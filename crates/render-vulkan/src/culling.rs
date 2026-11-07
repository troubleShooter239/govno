use crate::{
    compute::Compute,
    graph::Binding,
    memory::{Buffer, Memory},
    mesh::GpuMesh,
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{Material, MeshHandle, RenderExtent, RenderWorld, TextureHandle};
use gpu_allocator::MemoryLocation;
use std::{collections::HashMap, rc::Rc};

const INPUT_BYTES: u64 = 48;
pub(crate) const COMMAND_BYTES: u64 = 20;
type BatchKey = (MeshHandle, [Option<TextureHandle>; 5]);
pub(crate) struct Batch {
    pub mesh: MeshHandle,
    pub material: Material,
    pub base: u32,
    pub capacity: u32,
}
struct Slot {
    input: Buffer,
    commands: Buffer,
    counts: Buffer,
    readback: Buffer,
    capacity: usize,
    batches: Vec<Batch>,
    expected: Vec<u32>,
    objects: u32,
    push: Vec<u8>,
    pending: Option<(u64, bool)>,
    groups: HashMap<BatchKey, usize>,
    assignments: Vec<usize>,
    bytes: Vec<u8>,
    samples: Vec<(govno_render::Aabb, usize)>,
}
impl Slot {
    fn new(memory: Rc<Memory>, capacity: usize) -> Result<Self, BoxError> {
        use vk::BufferUsageFlags as U;
        Ok(Self {
            input: Buffer::new(
                memory.clone(),
                capacity as u64 * INPUT_BYTES,
                U::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?,
            commands: Buffer::new(
                memory.clone(),
                capacity as u64 * COMMAND_BYTES,
                U::STORAGE_BUFFER | U::INDIRECT_BUFFER,
                MemoryLocation::GpuOnly,
            )?,
            counts: Buffer::new(
                memory.clone(),
                capacity as u64 * 4,
                U::STORAGE_BUFFER | U::INDIRECT_BUFFER | U::TRANSFER_DST | U::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
            )?,
            readback: Buffer::new(
                memory,
                capacity as u64 * 8,
                U::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )?,
            capacity,
            batches: vec![],
            expected: vec![],
            objects: 0,
            push: vec![],
            pending: None,
            groups: HashMap::new(),
            assignments: vec![],
            bytes: vec![],
            samples: vec![],
        })
    }
}
pub(crate) struct Culling {
    pub hiz: Option<crate::hiz::HiZ>,
    compute: Compute,
    memory: Rc<Memory>,
    slots: Vec<Slot>,
    verify: bool,
    verified_frames: u64,
    latest: Option<(u64, u32, u32)>,
    occluded: u64,
}
impl Culling {
    pub fn new(
        memory: Rc<Memory>,
        frames: u32,
        verify: bool,
        layout: vk::DescriptorSetLayout,
    ) -> Result<Self, BoxError> {
        let bindings = (0..5)
            .map(|i| (i, vk::DescriptorType::STORAGE_BUFFER))
            .collect::<Vec<_>>();
        let compute = Compute::new(
            memory.ctx.clone(),
            include_bytes!("../shaders/cullScene.spv"),
            c"cullScene",
            &bindings,
            96,
            frames,
        )?;
        let slots = (0..frames)
            .map(|_| Slot::new(memory.clone(), 64))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            hiz: if memory.ctx.hiz {
                Some(crate::hiz::HiZ::new(
                    memory.clone(),
                    frames,
                    layout,
                    verify,
                )?)
            } else {
                None
            },
            compute,
            memory,
            slots,
            verify,
            verified_frames: 0,
            latest: None,
            occluded: 0,
        })
    }
    /// Only the completed frame slot may be prepared, resized or rebound.
    pub fn prepare(
        &mut self,
        frame: usize,
        world: &RenderWorld,
        extent: RenderExtent,
        meshes: &HashMap<MeshHandle, GpuMesh>,
        objects: Binding,
        depth: &crate::memory::ImageResource,
    ) -> Result<(), BoxError> {
        let ctx = &self.memory.ctx;
        let n = world.instances.len();
        if n as u64 > u64::from(ctx.capabilities.max_storage_buffer_range) / INPUT_BYTES
            || n.div_ceil(64) > ctx.capabilities.max_compute_groups as usize
        {
            return Err("scene exceeds compute culling buffer/dispatch limits".into());
        }
        let slot = &mut self.slots[frame];
        if slot.pending.is_some() {
            return Err("culling readback was not collected before slot reuse".into());
        }
        if n > slot.capacity {
            let capacity = n
                .next_power_of_two()
                .min(ctx.capabilities.max_storage_buffer_range as usize / INPUT_BYTES as usize);
            *slot = Slot::new(self.memory.clone(), capacity)?;
        }
        slot.batches.clear();
        slot.expected.clear();
        slot.samples.clear();
        slot.objects = n as u32;
        slot.groups.clear();
        slot.assignments.clear();
        slot.bytes.clear();
        for instance in &world.instances {
            let m = instance.material;
            let textures = if ctx.bindless {
                [None; 5]
            } else {
                [
                    m.base_color_texture,
                    m.normal_texture,
                    m.metallic_roughness_texture,
                    m.emissive_texture,
                    m.occlusion_texture,
                ]
            };
            let key = (instance.mesh, textures);
            let batch = match slot.groups.get(&key).copied() {
                Some(i) if slot.batches[i].capacity < ctx.capabilities.max_draw_indirect_count => i,
                _ => {
                    let i = slot.batches.len();
                    slot.batches.push(Batch {
                        mesh: instance.mesh,
                        material: m,
                        base: 0,
                        capacity: 0,
                    });
                    slot.groups.insert(key, i);
                    i
                }
            };
            slot.batches[batch].capacity += 1;
            slot.assignments.push(batch);
        }
        let mut base = 0;
        for batch in &mut slot.batches {
            batch.base = base;
            base += batch.capacity;
        }
        slot.expected.resize(slot.batches.len(), 0);
        let matrix = world.camera.view_projection(extent)?;
        let frustum = if self.verify {
            Some(govno_render::Frustum::from_view_projection(matrix)?)
        } else {
            None
        };
        slot.bytes.reserve(n * INPUT_BYTES as usize);
        for (id, (instance, batch_id)) in world
            .instances
            .iter()
            .zip(slot.assignments.iter().copied())
            .enumerate()
        {
            let mesh = meshes
                .get(&instance.mesh)
                .ok_or("stale culling mesh handle")?;
            for bound in [mesh.bounds.min, mesh.bounds.max] {
                for value in bound.to_array().into_iter().chain([0.0]) {
                    slot.bytes.extend(value.to_le_bytes());
                }
            }
            for value in [
                id as u32,
                mesh.index_count,
                slot.batches[batch_id].base,
                batch_id as u32,
            ] {
                slot.bytes.extend(value.to_le_bytes());
            }
            if let Some(frustum) = &frustum {
                let visible = !world.frustum_culling
                    || mesh
                        .bounds
                        .transformed(instance.transform.matrix())
                        .is_none_or(|bounds| frustum.intersects(bounds));
                slot.expected[batch_id] += u32::from(visible);
                if visible
                    && let Some(bounds) = mesh.bounds.transformed(instance.transform.matrix())
                {
                    slot.samples.push((bounds, batch_id));
                }
            }
        }
        slot.input.write(&slot.bytes)?;
        slot.push.clear();
        for value in matrix.to_cols_array() {
            slot.push.extend(value.to_le_bytes());
        }
        for value in [n as u32, u32::from(world.frustum_culling), 0, 0] {
            slot.push.extend(value.to_le_bytes());
        }
        for value in [extent.width, extent.height, 0, 0] {
            slot.push.extend(value.to_le_bytes());
        }
        if let Some(hiz) = &mut self.hiz {
            hiz.prepare(frame, extent, depth)?;
        }
        let Binding::Buffer { handle, size } = objects else {
            return Err("GPU Scene objects must be a buffer".into());
        };
        let infos = [
            (slot.input.handle, slot.capacity as u64 * INPUT_BYTES),
            (handle, size),
            (slot.commands.handle, slot.capacity as u64 * COMMAND_BYTES),
            (slot.counts.handle, slot.capacity as u64 * 4),
        ]
        .map(|(buffer, range)| {
            [vk::DescriptorBufferInfo::default()
                .buffer(buffer)
                .range(range)]
        });
        let writes = std::array::from_fn::<_, 4, _>(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.compute.sets[frame])
                .dst_binding(i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i])
        });
        // SAFETY: this slot completed; all referenced buffers remain owned through submission.
        unsafe {
            ctx.device.update_descriptor_sets(&writes, &[]);
            let binding = self.hiz.as_ref().map_or(
                Binding::Buffer {
                    handle: slot.input.handle,
                    size: INPUT_BYTES,
                },
                |h| h.binding(frame),
            );
            let Binding::Buffer { handle, size } = binding else {
                unreachable!()
            };
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(handle)
                .range(size)];
            ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.compute.sets[frame])
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)],
                &[],
            );
        }
        Ok(())
    }
    pub fn bindings(&self, frame: usize) -> [Binding; 4] {
        let s = &self.slots[frame];
        [
            (s.input.handle, s.capacity as u64 * INPUT_BYTES),
            (s.commands.handle, s.capacity as u64 * COMMAND_BYTES),
            (s.counts.handle, s.capacity as u64 * 4),
            (s.readback.handle, s.capacity as u64 * 8),
        ]
        .map(|(handle, size)| Binding::Buffer { handle, size })
    }
    pub fn reset(&self, cmd: vk::CommandBuffer, frame: usize) {
        let s = &self.slots[frame];
        if s.objects == 0 {
            return;
        }
        // SAFETY: owned transfer-destination buffer, aligned nonzero range within allocation.
        unsafe {
            self.memory.ctx.device.cmd_fill_buffer(
                cmd,
                s.counts.handle,
                0,
                s.batches.len() as u64 * 4,
                0,
            );
        }
    }
    pub fn dispatch(&self, cmd: vk::CommandBuffer, frame: usize, occlusion: bool) {
        let s = &self.slots[frame];
        if s.objects > 0 {
            let mut push = s.push.clone();
            push[72..76].copy_from_slice(&u32::from(occlusion).to_le_bytes());
            self.compute
                .dispatch(cmd, frame, &push, [s.objects.div_ceil(64), 1, 1]);
        }
    }
    pub fn batches(&self, frame: usize) -> &[Batch] {
        &self.slots[frame].batches
    }
    pub fn buffers(&self, frame: usize) -> (vk::Buffer, vk::Buffer) {
        let s = &self.slots[frame];
        (s.commands.handle, s.counts.handle)
    }
    pub fn copy_counts(&self, cmd: vk::CommandBuffer, frame: usize) {
        let s = &self.slots[frame];
        if s.objects == 0 {
            return;
        }
        // SAFETY: nonoverlapping owned buffers; graph orders indirect/compute uses and host readback.
        unsafe {
            self.memory.ctx.device.cmd_copy_buffer(
                cmd,
                s.counts.handle,
                s.readback.handle,
                &[vk::BufferCopy::default().size(s.batches.len() as u64 * 4)],
            );
        }
    }
    pub fn copy_candidates(&self, cmd: vk::CommandBuffer, frame: usize) {
        let s = &self.slots[frame];
        if s.objects == 0 {
            return;
        }
        // SAFETY: graph orders the first culling result before reset; readback regions do not overlap.
        unsafe {
            self.memory.ctx.device.cmd_copy_buffer(
                cmd,
                s.counts.handle,
                s.readback.handle,
                &[vk::BufferCopy::default()
                    .dst_offset(s.capacity as u64 * 4)
                    .size(s.batches.len() as u64 * 4)],
            );
        }
    }
    pub fn depth_prepass(
        &self,
        cmd: vk::CommandBuffer,
        frame: usize,
        extent: vk::Extent2D,
        set: vk::DescriptorSet,
        meshes: &HashMap<MeshHandle, GpuMesh>,
    ) -> Result<(), BoxError> {
        let hiz = self.hiz.as_ref().ok_or("Hi-Z depth pipeline missing")?;
        let s = &self.slots[frame];
        let device = &self.memory.ctx.device;
        // SAFETY: caller is recording the completed frame slot; all meshes, descriptors and depth image remain live.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, hiz.depth.pipeline);
            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                }],
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                hiz.depth.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                hiz.depth.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &s.push[..64],
            );
            for (id, batch) in s.batches.iter().enumerate() {
                let mesh = meshes.get(&batch.mesh).ok_or("stale depth-prepass mesh")?;
                device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertices.handle], &[0]);
                device.cmd_bind_index_buffer(cmd, mesh.indices.handle, 0, vk::IndexType::UINT32);
                device.cmd_draw_indexed_indirect_count(
                    cmd,
                    s.commands.handle,
                    u64::from(batch.base) * COMMAND_BYTES,
                    s.counts.handle,
                    id as u64 * 4,
                    batch.capacity,
                    COMMAND_BYTES as u32,
                );
            }
        }
        Ok(())
    }
    pub fn submitted(&mut self, frame: usize, serial: u64, measured: bool) {
        self.slots[frame].pending = Some((serial, measured));
    }
    pub fn collect(
        &mut self,
        completed: u64,
        measurements: &mut crate::metrics::Measurements,
    ) -> Result<Option<(u32, u32)>, BoxError> {
        for (frame, s) in self.slots.iter_mut().enumerate() {
            let Some((serial, measured)) = s.pending else {
                continue;
            };
            if serial > completed {
                continue;
            }
            let bytes = if s.objects > 0 {
                s.readback.read()?
            } else {
                vec![]
            };
            let actual = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .take(s.batches.len())
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>();
            if actual.iter().zip(&s.batches).any(|(&n, b)| n > b.capacity) {
                return Err("GPU culling counter exceeds batch capacity".into());
            }
            let candidates = if self.hiz.is_some() {
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .skip(s.capacity)
                    .take(s.batches.len())
                    .map(|b| u32::from_le_bytes(*b))
                    .collect::<Vec<_>>()
            } else {
                actual.clone()
            };
            if candidates
                .iter()
                .zip(&s.batches)
                .any(|(&n, b)| n > b.capacity)
            {
                return Err("frustum counter exceeds batch capacity".into());
            }
            if actual.iter().zip(&candidates).any(|(&a, &b)| a > b) {
                return Err("occlusion increased visible counts".into());
            }
            if self.verify {
                if let Some((batch, (&gpu, &cpu))) = candidates
                    .iter()
                    .zip(&s.expected)
                    .enumerate()
                    .find(|(_, (gpu, cpu))| gpu != cpu)
                {
                    return Err(format!(
                        "GPU/CPU visibility mismatch at serial {serial}, batch {batch}: GPU {gpu}, CPU {cpu}"
                    )
                    .into());
                }
                if let Some(hiz) = &mut self.hiz {
                    let data = hiz.read(frame)?;
                    let values: [f32; 16] = std::array::from_fn(|i| {
                        f32::from_le_bytes(s.push[i * 4..i * 4 + 4].try_into().unwrap())
                    });
                    let matrix = glam::Mat4::from_cols_array(&values);
                    let mut required = s.expected.clone();
                    let mut allowed = s.expected.clone();
                    for &(bounds, batch) in &s.samples {
                        if hiz.occluded(frame, &data, bounds, matrix, -2e-6) {
                            required[batch] -= 1;
                        }
                        if hiz.occluded(frame, &data, bounds, matrix, 2e-6) {
                            allowed[batch] -= 1;
                        }
                    }
                    for (batch, &gpu) in actual.iter().enumerate() {
                        if gpu < required[batch] || gpu > allowed[batch] {
                            return Err(format!("Hi-Z visibility mismatch at serial {serial}, batch {batch}: GPU {gpu}, CPU {}..={}",required[batch],allowed[batch]).into());
                        }
                    }
                }
                self.verified_frames += 1;
            }
            let visible = actual.iter().sum::<u32>();
            let culled = s.objects - visible;
            let occluded = candidates.iter().sum::<u32>() - visible;
            self.occluded += u64::from(occluded);
            if measured {
                measurements.submitted += u64::from(visible);
                measurements.culled += u64::from(culled);
                measurements.occluded += u64::from(occluded);
            }
            if self.latest.is_none_or(|(previous, _, _)| serial > previous) {
                self.latest = Some((serial, visible, culled));
            }
            s.pending = None;
        }
        Ok(self.latest.map(|(_, visible, culled)| (visible, culled)))
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"enabled":true,"verification":self.verify,"verified_frames":self.verified_frames,"occluded":self.occluded,"hiz":self.hiz.as_ref().map(|h|h.report()),"capacities":self.slots.iter().map(|s|s.capacity).collect::<Vec<_>>(),"batches":self.slots.iter().map(|s|s.batches.len()).collect::<Vec<_>>()})
    }
}

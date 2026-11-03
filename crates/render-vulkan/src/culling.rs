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
                capacity as u64 * 4,
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
        })
    }
}
pub(crate) struct Culling {
    compute: Compute,
    memory: Rc<Memory>,
    slots: Vec<Slot>,
    verify: bool,
    verified_frames: u64,
    latest: Option<(u64, u32, u32)>,
}
impl Culling {
    pub fn new(memory: Rc<Memory>, frames: u32, verify: bool) -> Result<Self, BoxError> {
        let bindings = (0..4)
            .map(|i| (i, vk::DescriptorType::STORAGE_BUFFER))
            .collect::<Vec<_>>();
        let compute = Compute::new(
            memory.ctx.clone(),
            include_bytes!("../shaders/cullScene.spv"),
            c"cullScene",
            &bindings,
            80,
            frames,
        )?;
        let slots = (0..frames)
            .map(|_| Slot::new(memory.clone(), 64))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            compute,
            memory,
            slots,
            verify,
            verified_frames: 0,
            latest: None,
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
            }
        }
        slot.input.write(&slot.bytes)?;
        slot.push.clear();
        for value in matrix.to_cols_array() {
            slot.push.extend(value.to_le_bytes());
        }
        for value in [
            n as u32,
            u32::from(world.frustum_culling),
            slot.batches.len() as u32,
            0,
        ] {
            slot.push.extend(value.to_le_bytes());
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
        }
        Ok(())
    }
    pub fn bindings(&self, frame: usize) -> [Binding; 4] {
        let s = &self.slots[frame];
        [
            (s.input.handle, s.capacity as u64 * INPUT_BYTES),
            (s.commands.handle, s.capacity as u64 * COMMAND_BYTES),
            (s.counts.handle, s.capacity as u64 * 4),
            (s.readback.handle, s.capacity as u64 * 4),
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
    pub fn dispatch(&self, cmd: vk::CommandBuffer, frame: usize) {
        let s = &self.slots[frame];
        if s.objects > 0 {
            self.compute
                .dispatch(cmd, frame, &s.push, [s.objects.div_ceil(64), 1, 1]);
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
    pub fn submitted(&mut self, frame: usize, serial: u64, measured: bool) {
        self.slots[frame].pending = Some((serial, measured));
    }
    pub fn collect(
        &mut self,
        completed: u64,
        measurements: &mut crate::metrics::Measurements,
    ) -> Result<Option<(u32, u32)>, BoxError> {
        for s in &mut self.slots {
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
            if self.verify {
                if let Some((batch, (&gpu, &cpu))) = actual
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
                self.verified_frames += 1;
            }
            let visible = actual.iter().sum::<u32>();
            let culled = s.objects - visible;
            if measured {
                measurements.submitted += u64::from(visible);
                measurements.culled += u64::from(culled);
            }
            if self.latest.is_none_or(|(previous, _, _)| serial > previous) {
                self.latest = Some((serial, visible, culled));
            }
            s.pending = None;
        }
        Ok(self.latest.map(|(_, visible, culled)| (visible, culled)))
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"enabled":true,"verification":self.verify,"verified_frames":self.verified_frames,"capacities":self.slots.iter().map(|s|s.capacity).collect::<Vec<_>>(),"batches":self.slots.iter().map(|s|s.batches.len()).collect::<Vec<_>>()})
    }
}

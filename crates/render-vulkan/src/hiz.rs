use crate::{
    compute::Compute,
    graph::Binding,
    memory::{Buffer, ImageResource, Memory},
    pipeline::ForwardPipeline,
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{Aabb, RenderExtent};
use gpu_allocator::MemoryLocation;
use std::rc::Rc;

struct Slot {
    buffer: Buffer,
    size: u64,
    extent: RenderExtent,
    levels: Vec<[u32; 3]>,
}
pub(crate) struct HiZ {
    memory: Rc<Memory>,
    compute: Compute,
    pub depth: ForwardPipeline,
    slots: Vec<Slot>,
    verify: bool,
}
impl HiZ {
    pub fn new(
        memory: Rc<Memory>,
        frames: u32,
        layout: vk::DescriptorSetLayout,
        verify: bool,
    ) -> Result<Self, BoxError> {
        let compute = Compute::new(
            memory.ctx.clone(),
            include_bytes!("../shaders/reduceDepth.spv"),
            c"reduceDepth",
            &[
                (0, vk::DescriptorType::SAMPLED_IMAGE),
                (2, vk::DescriptorType::STORAGE_BUFFER),
            ],
            32,
            frames,
        )?;
        let depth = ForwardPipeline::depth(memory.ctx.clone(), layout)?;
        let slots = (0..frames)
            .map(|_| {
                Self::slot(
                    memory.clone(),
                    RenderExtent {
                        width: 1,
                        height: 1,
                    },
                    verify,
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            memory,
            compute,
            depth,
            slots,
            verify,
        })
    }
    fn slot(memory: Rc<Memory>, extent: RenderExtent, verify: bool) -> Result<Slot, BoxError> {
        let (mut w, mut h, mut offset) = (extent.width, extent.height, 0u64);
        let mut levels = vec![];
        loop {
            levels.push([offset as u32, w, h]);
            offset += u64::from(w) * u64::from(h);
            if w == 1 && h == 1 {
                break;
            }
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        let size = offset * 4;
        if size > u64::from(memory.ctx.capabilities.max_storage_buffer_range) {
            return Err("Hi-Z pyramid exceeds storage-buffer limit".into());
        }
        let buffer = Buffer::new(
            memory,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            if verify {
                MemoryLocation::CpuToGpu
            } else {
                MemoryLocation::GpuOnly
            },
        )?;
        Ok(Slot {
            buffer,
            size,
            extent,
            levels,
        })
    }
    pub fn prepare(
        &mut self,
        frame: usize,
        extent: RenderExtent,
        depth: &ImageResource,
    ) -> Result<(), BoxError> {
        let s = &self.slots[frame];
        if s.extent.width != extent.width || s.extent.height != extent.height {
            self.slots[frame] = Self::slot(self.memory.clone(), extent, self.verify)?;
        }
        let s = &self.slots[frame];
        let image = [vk::DescriptorImageInfo::default()
            .image_view(depth.view)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
        let buffer = [vk::DescriptorBufferInfo::default()
            .buffer(s.buffer.handle)
            .range(s.size)];
        // SAFETY: frame slot completed; descriptor resources remain owned through submission.
        unsafe {
            self.memory.ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.compute.sets[frame])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&image),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.compute.sets[frame])
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer),
                ],
                &[],
            );
        }
        Ok(())
    }
    pub fn binding(&self, frame: usize) -> Binding {
        let s = &self.slots[frame];
        Binding::Buffer {
            handle: s.buffer.handle,
            size: s.size,
        }
    }
    pub fn record(&self, cmd: vk::CommandBuffer, frame: usize) {
        let s = &self.slots[frame];
        for (level, &[offset, w, h]) in s.levels.iter().enumerate() {
            let [previous, pw, ph] = if level == 0 {
                [0, w, h]
            } else {
                s.levels[level - 1]
            };
            let push = [previous, offset, pw, ph, w, h, u32::from(level == 0), 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            if level > 0 {
                // SAFETY: adjacent reductions share one owned buffer; each level consumes the preceding writes.
                unsafe {
                    self.memory.ctx.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&[
                            vk::MemoryBarrier2::default()
                                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                                .dst_access_mask(
                                    vk::AccessFlags2::SHADER_STORAGE_READ
                                        | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                                ),
                        ]),
                    );
                }
            }
            self.compute
                .dispatch(cmd, frame, &push, [w.div_ceil(8), h.div_ceil(8), 1]);
        }
    }
    pub fn read(&mut self, frame: usize) -> Result<Vec<f32>, BoxError> {
        let s = &mut self.slots[frame];
        let bytes = s.buffer.read()?;
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        if data
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("Hi-Z contains invalid depth".into());
        }
        for level in 1..s.levels.len() {
            let [src, sw, sh] = s.levels[level - 1];
            let [dst, w, h] = s.levels[level];
            for y in 0..h {
                for x in 0..w {
                    let mut minimum = 1.0f32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (px, py) = (x * 2 + dx, y * 2 + dy);
                            if px < sw && py < sh {
                                minimum = minimum.min(data[(src + py * sw + px) as usize]);
                            }
                        }
                    }
                    if data[(dst + y * w + x) as usize] != minimum {
                        return Err(
                            format!("Hi-Z reduction mismatch at level {level}, {x},{y}").into()
                        );
                    }
                }
            }
        }
        Ok(data)
    }
    pub fn occluded(
        &self,
        frame: usize,
        data: &[f32],
        bounds: Aabb,
        matrix: glam::Mat4,
        margin: f32,
    ) -> bool {
        let s = &self.slots[frame];
        let mut minimum = glam::Vec2::splat(f32::INFINITY);
        let mut maximum = glam::Vec2::splat(f32::NEG_INFINITY);
        let mut nearest = 0.0f32;
        for x in [bounds.min.x, bounds.max.x] {
            for y in [bounds.min.y, bounds.max.y] {
                for z in [bounds.min.z, bounds.max.z] {
                    let p = matrix * glam::Vec4::new(x, y, z, 1.0);
                    if !p.is_finite() || p.w <= 0.0 || p.z <= 0.0 || p.z >= p.w {
                        return false;
                    }
                    let ndc = p.truncate() / p.w;
                    minimum = minimum.min(ndc.truncate());
                    maximum = maximum.max(ndc.truncate());
                    nearest = nearest.max(ndc.z);
                }
            }
        }
        if minimum.min_element() <= -1.0 || maximum.max_element() >= 1.0 {
            return false;
        }
        let size = glam::Vec2::new(s.extent.width as f32, s.extent.height as f32);
        let lo = ((minimum * 0.5 + glam::Vec2::splat(0.5)) * size - glam::Vec2::ONE)
            .floor()
            .max(glam::Vec2::ZERO)
            .as_uvec2();
        let hi = ((maximum * 0.5 + glam::Vec2::splat(0.5)) * size + glam::Vec2::ONE)
            .ceil()
            .min(size - glam::Vec2::ONE)
            .as_uvec2();
        let mut level = 0u32;
        while ((hi >> level) - (lo >> level)).max_element() > 1 && level + 1 < s.levels.len() as u32
        {
            level += 1;
        }
        let [offset, w, _] = s.levels[level as usize];
        let a = lo >> level;
        let b = hi >> level;
        let mut depth = 1.0f32;
        for y in a.y..=b.y {
            for x in a.x..=b.x {
                depth = depth.min(data[(offset + y * w + x) as usize]);
            }
        }
        nearest + margin + 1e-5f32.max(nearest * 1e-4) < depth
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"same_frame":true,"buffer_bytes":self.slots.iter().map(|s|s.size).collect::<Vec<_>>(),"mip_levels":self.slots.iter().map(|s|s.levels.len()).collect::<Vec<_>>()})
    }
}

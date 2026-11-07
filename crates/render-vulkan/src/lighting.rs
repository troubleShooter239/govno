use crate::{
    compute::Compute,
    graph::Binding,
    memory::{Buffer, Memory},
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{RenderExtent, RenderWorld};
use gpu_allocator::MemoryLocation;
use std::rc::Rc;

pub(crate) const LIGHTS: usize = govno_render::MAX_POINT_LIGHTS;
const MASK_WORDS: usize = LIGHTS / 32;
const MASK_OFFSET: usize = 4 + LIGHTS * 8;
struct Slot {
    buffer: Buffer,
    size: u64,
    tiles: [u32; 2],
    push: Vec<u8>,
    expected: Vec<(u32, u32)>,
    pending: Option<u64>,
}
pub(crate) struct Lighting {
    memory: Rc<Memory>,
    compute: Option<Compute>,
    slots: Vec<Slot>,
    bytes: Vec<u8>,
    verified_frames: u64,
    verified_references: u64,
    total_references: u64,
}
impl Lighting {
    pub fn new(memory: Rc<Memory>, frames: u32) -> Result<Self, BoxError> {
        let compute = if memory.ctx.forward_plus {
            Some(Compute::new(
                memory.ctx.clone(),
                include_bytes!("../shaders/cullLights.spv"),
                c"cullLights",
                &[(0, vk::DescriptorType::STORAGE_BUFFER)],
                80,
                frames,
            )?)
        } else {
            None
        };
        let slots = (0..frames)
            .map(|_| Self::slot(memory.clone(), [1, 1]))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            memory,
            compute,
            slots,
            bytes: Vec::with_capacity(MASK_OFFSET * 4),
            verified_frames: 0,
            verified_references: 0,
            total_references: 0,
        })
    }
    fn slot(memory: Rc<Memory>, tiles: [u32; 2]) -> Result<Slot, BoxError> {
        let size = (MASK_OFFSET as u64
            + u64::from(tiles[0]) * u64::from(tiles[1]) * MASK_WORDS as u64)
            * 4;
        if size > u64::from(memory.ctx.capabilities.max_storage_buffer_range) {
            return Err("Forward+ tile buffer exceeds maxStorageBufferRange".into());
        }
        let buffer = Buffer::new(
            memory,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::CpuToGpu,
        )?;
        Ok(Slot {
            buffer,
            size,
            tiles,
            push: Vec::with_capacity(80),
            expected: vec![],
            pending: None,
        })
    }
    /// Caller waited for the frame slot; returns whether its descriptor changed.
    pub fn update(
        &mut self,
        frame: usize,
        world: &RenderWorld,
        extent: RenderExtent,
    ) -> Result<bool, BoxError> {
        let tiles = [extent.width.div_ceil(16), extent.height.div_ceil(16)];
        if self.slots[frame].pending.is_some() {
            return Err("lighting verification must complete before slot reuse".into());
        }
        let ctx = &self.memory.ctx;
        if tiles[0] > ctx.capabilities.max_compute_groups
            || tiles[1] > ctx.capabilities.max_compute_groups_y
        {
            return Err("Forward+ tile dispatch exceeds device limits".into());
        }
        let changed = self.slots[frame].tiles != tiles;
        if changed {
            self.slots[frame] = Self::slot(self.memory.clone(), tiles)?;
        }
        let slot = &mut self.slots[frame];
        if ctx.verify_lighting {
            slot.expected = reference(world, extent, tiles)?;
        }
        self.bytes.clear();
        for value in [
            tiles[0],
            tiles[1],
            MASK_WORDS as u32,
            u32::from(self.compute.is_some()),
        ] {
            self.bytes.extend(value.to_le_bytes());
        }
        for light in &world.point_lights {
            for value in light
                .position
                .extend(light.range)
                .to_array()
                .into_iter()
                .chain(light.intensity.extend(0.0).to_array())
            {
                self.bytes.extend(value.to_le_bytes());
            }
        }
        self.bytes.resize(MASK_OFFSET * 4, 0);
        slot.buffer.write(&self.bytes)?;
        slot.push.clear();
        for value in world.camera.view_projection(extent)?.to_cols_array() {
            slot.push.extend(value.to_le_bytes());
        }
        for value in [
            extent.width,
            extent.height,
            world.point_lights.len() as u32,
            0,
        ] {
            slot.push.extend(value.to_le_bytes());
        }
        if let Some(compute) = &self.compute {
            self.bind(compute.sets[frame], 0, frame);
        }
        Ok(changed)
    }
    pub fn bind(&self, set: vk::DescriptorSet, binding: u32, frame: usize) {
        let slot = &self.slots[frame];
        let info = [vk::DescriptorBufferInfo::default()
            .buffer(slot.buffer.handle)
            .range(slot.size)];
        // SAFETY: caller only updates new or completed descriptor sets; slot owns the buffer.
        unsafe {
            self.memory.ctx.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)],
                &[],
            );
        }
    }
    pub fn binding(&self, frame: usize) -> Binding {
        let s = &self.slots[frame];
        Binding::Buffer {
            handle: s.buffer.handle,
            size: s.size,
        }
    }
    pub fn record(&self, cmd: vk::CommandBuffer, frame: usize) {
        if let Some(c) = &self.compute {
            let s = &self.slots[frame];
            c.dispatch(cmd, frame, &s.push, [s.tiles[0], s.tiles[1], 1]);
        }
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"enabled":self.compute.is_some(),"max_lights":LIGHTS,"tile_size":16,"mask_words":MASK_WORDS,"frame_tiles":self.slots.iter().map(|s|s.tiles).collect::<Vec<_>>(),"buffer_bytes":self.slots.iter().map(|s|s.size).collect::<Vec<_>>(),"verification":{"enabled":self.memory.ctx.verify_lighting,"frames":self.verified_frames,"tile_light_references":self.verified_references,"unculled_references":self.total_references}})
    }
    pub fn submitted(&mut self, frame: usize, serial: u64) {
        if self.memory.ctx.verify_lighting {
            self.slots[frame].pending = Some(serial);
        }
    }
    pub fn collect(&mut self, completed: u64) -> Result<(), BoxError> {
        for slot in &mut self.slots {
            let Some(serial) = slot.pending else {
                continue;
            };
            if serial > completed {
                continue;
            }
            let bytes = slot.buffer.read()?;
            let words = bytes.as_chunks::<4>().0;
            for (index, &(required, allowed)) in slot.expected.iter().enumerate() {
                let actual = u32::from_le_bytes(words[MASK_OFFSET + index]);
                if (actual & required) != required || (actual & !allowed) != 0 {
                    return Err(format!("Forward+ mask mismatch at serial {serial}, tile {}, word {}: GPU {actual:08x}, required {required:08x}, allowed {allowed:08x}",index/MASK_WORDS,index%MASK_WORDS).into());
                }
                self.verified_references += u64::from(actual.count_ones());
            }
            let lights = u32::from_le_bytes(slot.push[72..76].try_into()?);
            self.total_references +=
                u64::from(lights) * u64::from(slot.tiles[0]) * u64::from(slot.tiles[1]);
            self.verified_frames += 1;
            slot.pending = None;
        }
        Ok(())
    }
}

fn reference(
    world: &RenderWorld,
    extent: RenderExtent,
    tiles: [u32; 2],
) -> Result<Vec<(u32, u32)>, BoxError> {
    use glam::DVec4;
    let rows = world
        .camera
        .view_projection(extent)?
        .transpose()
        .to_cols_array_2d()
        .map(|v| DVec4::from_array(v.map(f64::from)));
    let mut masks = vec![(0, 0); tiles[0] as usize * tiles[1] as usize * MASK_WORDS];
    for y in 0..tiles[1] {
        for x in 0..tiles[0] {
            let lower = [
                f64::from(x * 16) / f64::from(extent.width) * 2.0 - 1.0,
                f64::from(y * 16) / f64::from(extent.height) * 2.0 - 1.0,
            ];
            let upper = [
                f64::from(((x + 1) * 16).min(extent.width)) / f64::from(extent.width) * 2.0 - 1.0,
                f64::from(((y + 1) * 16).min(extent.height)) / f64::from(extent.height) * 2.0 - 1.0,
            ];
            let planes = [
                rows[0] - lower[0] * rows[3],
                upper[0] * rows[3] - rows[0],
                rows[1] - lower[1] * rows[3],
                upper[1] * rows[3] - rows[1],
                rows[2],
                rows[3] - rows[2],
            ];
            for (id, light) in world.point_lights.iter().enumerate() {
                let center = light.position.as_dvec3().extend(1.0);
                let mut required = true;
                let mut allowed = true;
                for p in planes {
                    let radius = f64::from(light.range) * p.truncate().length();
                    let distance = p.dot(center) + radius;
                    let scale = (p.abs().dot(center.abs()) + radius).max(1.0);
                    // Permit the float GPU's rounding band around the conservative sphere test.
                    required &= distance >= -8e-6 * scale;
                    allowed &= distance >= -1.2e-5 * scale;
                }
                let word = &mut masks[((y * tiles[0] + x) as usize) * MASK_WORDS + id / 32];
                if required {
                    word.0 |= 1 << (id % 32);
                }
                if allowed {
                    word.1 |= 1 << (id % 32);
                }
            }
        }
    }
    Ok(masks)
}

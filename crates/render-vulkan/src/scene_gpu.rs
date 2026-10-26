use crate::{
    context::DeviceContext,
    memory::{Buffer, Memory},
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{DrawInstance, RenderExtent, RenderWorld};
use gpu_allocator::MemoryLocation;
use std::{rc::Rc, sync::Arc};

pub(crate) const CAMERA_BYTES: u64 = 112;
pub(crate) const DRAW_BYTES: u32 = 96;
pub(crate) struct SceneGpu {
    ctx: Arc<DeviceContext>,
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    cameras: Vec<Buffer>,
}
impl SceneGpu {
    pub fn new(memory: Rc<Memory>, frames: u32) -> Result<Self, BoxError> {
        let mut owner = Self {
            ctx: memory.ctx.clone(),
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            cameras: Vec::new(),
        };
        for _ in 0..frames {
            owner.cameras.push(Buffer::new(
                memory.clone(),
                CAMERA_BYTES,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
        }
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)];
        // SAFETY: layouts and buffers are kept alive by owner. Descriptor updates
        // occur only at startup, before any queue references these sets.
        unsafe {
            owner.layout = owner.ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(frames)];
            owner.pool = owner.ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(frames)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = vec![owner.layout; frames as usize];
            owner.sets = owner.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(owner.pool)
                    .set_layouts(&layouts),
            )?;
            for (set, buffer) in owner.sets.iter().zip(&owner.cameras) {
                let info = [vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .range(CAMERA_BYTES)];
                owner.ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&info)],
                    &[],
                );
            }
        }
        Ok(owner)
    }
    /// Call only after waiting for this frame slot's submission fence.
    pub fn update(
        &mut self,
        frame: usize,
        world: &RenderWorld,
        extent: RenderExtent,
        srgb_attachment: bool,
    ) -> Result<(), BoxError> {
        let mut data = world
            .camera
            .view_projection(extent)?
            .to_cols_array()
            .to_vec();
        data.extend_from_slice(&world.camera.position.extend(1.0).to_array());
        data.extend_from_slice(&world.light_direction.normalize().extend(0.0).to_array());
        data.extend_from_slice(
            &world
                .light_radiance
                .extend(if srgb_attachment { 0.0 } else { 1.0 })
                .to_array(),
        );
        self.cameras[frame].write(&floats(&data))
    }
}
impl Drop for SceneGpu {
    fn drop(&mut self) {
        // SAFETY: renderer has waited for all referencing submissions and dropped
        // pipelines before these shared bindings. Pool owns its descriptor sets.
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
pub(crate) fn draw_constants(instance: &DrawInstance) -> Vec<u8> {
    let mut data = instance.transform.matrix().to_cols_array().to_vec();
    let inverse_scale_squared = (instance.transform.scale * instance.transform.scale).recip();
    data.extend_from_slice(
        &inverse_scale_squared
            .extend(instance.material.roughness)
            .to_array(),
    );
    data.extend_from_slice(
        &instance
            .material
            .base_color
            .extend(instance.material.metallic)
            .to_array(),
    );
    floats(&data)
}
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use govno_render::{Material, MeshHandle, Transform};
    #[test]
    fn push_constant_layout_matches_shader_offsets() {
        let item = DrawInstance {
            mesh: MeshHandle::from_parts(1, 0),
            transform: Transform {
                scale: Vec3::new(2.0, 1.0, 1.0),
                ..Default::default()
            },
            material: Material {
                base_color: Vec3::ONE,
                metallic: 0.7,
                roughness: 0.4,
            },
        };
        let bytes = draw_constants(&item);
        assert_eq!(bytes.len(), DRAW_BYTES as usize);
        let value = |offset| f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
        assert_eq!(value(64), 0.25);
        assert_eq!(value(76), 0.4);
        assert_eq!(value(92), 0.7);
    }
}

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
pub(crate) const DRAW_BYTES: u32 = 128;
pub(crate) struct SceneGpu {
    ctx: Arc<DeviceContext>,
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    cameras: Vec<Buffer>,
    texture: Option<crate::texture::Texture>,
    extras: std::collections::HashMap<
        (
            Option<govno_render::TextureHandle>,
            Option<govno_render::TextureHandle>,
        ),
        ExtraSet,
    >,
    memory: Rc<Memory>,
    textures: std::collections::HashMap<govno_render::TextureHandle, TextureSlot>,
}
impl SceneGpu {
    pub fn new(memory: Rc<Memory>, frames: u32) -> Result<Self, BoxError> {
        let mut owner = Self {
            ctx: memory.ctx.clone(),
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            cameras: Vec::new(),
            texture: None,
            extras: Default::default(),
            memory: memory.clone(),
            textures: Default::default(),
        };
        for _ in 0..frames {
            owner.cameras.push(Buffer::new(
                memory.clone(),
                CAMERA_BYTES,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
        }
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        // SAFETY: layouts and buffers are kept alive by owner. Descriptor updates
        // occur only at startup, before any queue references these sets.
        unsafe {
            owner.layout = owner.ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(frames),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(frames * 2),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(frames * 2),
            ];
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
        owner.set_texture(&govno_render::TextureData {
            color_space: Default::default(),
            width: 1,
            height: 1,
            rgba8: vec![255; 4],
        })?;
        for (index, rgba8) in [(0, vec![128, 128, 255, 255]), (1, vec![255; 4])] {
            owner.upload_texture(
                govno_render::TextureHandle::from_parts(0, index),
                &govno_render::TextureData {
                    width: 1,
                    height: 1,
                    rgba8,
                    color_space: govno_render::TextureColorSpace::Linear,
                },
                Default::default(),
            )?;
        }
        owner.upload_texture(
            govno_render::TextureHandle::from_parts(0, 2),
            &govno_render::TextureData {
                width: 1,
                height: 1,
                rgba8: vec![255; 4],
                color_space: govno_render::TextureColorSpace::Srgb,
            },
            Default::default(),
        )?;
        Ok(owner)
    }
    /// Caller has quiesced all descriptor users before entering this method.
    pub fn set_texture(&mut self, data: &govno_render::TextureData) -> Result<(), BoxError> {
        let texture = crate::texture::Texture::new(self.memory.clone(), data)?;
        let image = [vk::DescriptorImageInfo::default()
            .image_view(texture.image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let sampler = [vk::DescriptorImageInfo::default().sampler(texture.sampler)];
        for set in &self.sets {
            // SAFETY: no pending descriptor users; new texture remains owned until
            // a future quiesced replacement or renderer teardown.
            unsafe {
                self.ctx.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&image),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler),
                    ],
                    &[],
                );
            }
        }
        self.texture = Some(texture);
        Ok(())
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
    data.extend_from_slice(&[
        instance.material.normal_scale,
        if instance.material.normal_texture.is_some() {
            1.0
        } else {
            0.0
        },
        0.0,
        0.0,
    ]);
    data.extend_from_slice(
        &instance
            .material
            .emissive
            .extend(instance.material.occlusion_strength)
            .to_array(),
    );
    floats(&data)
}
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

struct TextureSlot {
    ctx: Arc<DeviceContext>,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    _texture: crate::texture::Texture,
    color_space: govno_render::TextureColorSpace,
}
impl Drop for TextureSlot {
    fn drop(&mut self) {
        // SAFETY: renderer waits for users before removing slots or shutting down.
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
impl SceneGpu {
    pub fn upload_texture(
        &mut self,
        handle: govno_render::TextureHandle,
        data: &govno_render::TextureData,
        sampling: govno_render::TextureSampling,
    ) -> Result<(), BoxError> {
        let texture = crate::texture::Texture::with_sampling(self.memory.clone(), data, sampling)?;
        let frames = self.cameras.len() as u32;
        let mut slot = TextureSlot {
            ctx: self.ctx.clone(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            _texture: texture,
            color_space: data.color_space,
        };
        // SAFETY: new descriptor sets have no GPU users; retained cameras/textures
        // outlive the slot. Partial allocation is owned by its pool.
        unsafe {
            let sizes = [
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::SAMPLER,
            ]
            .map(|ty| {
                vk::DescriptorPoolSize::default().ty(ty).descriptor_count(
                    if ty == vk::DescriptorType::UNIFORM_BUFFER {
                        frames
                    } else {
                        frames * 2
                    },
                )
            });
            slot.pool = self.ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(frames)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = vec![self.layout; frames as usize];
            slot.sets = self.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(slot.pool)
                    .set_layouts(&layouts),
            )?;
            for (set, camera) in slot.sets.iter().zip(&self.cameras) {
                let buffer = [vk::DescriptorBufferInfo::default()
                    .buffer(camera.handle)
                    .range(CAMERA_BYTES)];
                let image = [vk::DescriptorImageInfo::default()
                    .image_view(slot._texture.image.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let sampler = [vk::DescriptorImageInfo::default().sampler(slot._texture.sampler)];
                self.ctx.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&buffer),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&image),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler),
                    ],
                    &[],
                );
            }
        }
        self.textures.insert(handle, slot);
        Ok(())
    }
    pub fn descriptor(
        &self,
        frame: usize,
        texture: Option<govno_render::TextureHandle>,
    ) -> Result<vk::DescriptorSet, BoxError> {
        match texture {
            None => Ok(self.sets[frame]),
            Some(handle) => Ok(self
                .textures
                .get(&handle)
                .ok_or("unknown or foreign texture handle")?
                .sets[frame]),
        }
    }
    pub fn descriptors(
        &self,
        frame: usize,
        material: &govno_render::Material,
    ) -> Result<[vk::DescriptorSet; 4], BoxError> {
        for (handle, space) in [
            (
                material.emissive_texture,
                govno_render::TextureColorSpace::Srgb,
            ),
            (
                material.occlusion_texture,
                govno_render::TextureColorSpace::Linear,
            ),
            (
                material.base_color_texture,
                govno_render::TextureColorSpace::Srgb,
            ),
            (
                material.normal_texture,
                govno_render::TextureColorSpace::Linear,
            ),
            (
                material.metallic_roughness_texture,
                govno_render::TextureColorSpace::Linear,
            ),
        ] {
            if let Some(handle) = handle
                && self
                    .textures
                    .get(&handle)
                    .ok_or("unknown texture handle")?
                    .color_space
                    != space
            {
                return Err("texture color space does not match material slot".into());
            }
        }
        Ok([
            self.descriptor(frame, material.base_color_texture)?,
            self.descriptor(
                frame,
                Some(
                    material
                        .normal_texture
                        .unwrap_or(govno_render::TextureHandle::from_parts(0, 0)),
                ),
            )?,
            self.descriptor(
                frame,
                Some(
                    material
                        .metallic_roughness_texture
                        .unwrap_or(govno_render::TextureHandle::from_parts(0, 1)),
                ),
            )?,
            self.extras
                .get(&(material.emissive_texture, material.occlusion_texture))
                .ok_or("material descriptors not prepared")?
                .sets[frame],
        ])
    }
    pub fn remove_texture(&mut self, handle: govno_render::TextureHandle) -> Result<(), BoxError> {
        if handle == govno_render::TextureHandle::from_parts(0, 2)
            || handle == govno_render::TextureHandle::from_parts(0, 0)
            || handle == govno_render::TextureHandle::from_parts(0, 1)
        {
            return Err("reserved fallback texture".into());
        }
        self.extras.clear();
        self.textures
            .remove(&handle)
            .ok_or("unknown or removed texture handle")?;
        Ok(())
    }
}

struct ExtraSet {
    ctx: Arc<DeviceContext>,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}
impl Drop for ExtraSet {
    fn drop(&mut self) {
        // SAFETY: immutable sets are released only after renderer quiescence.
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
impl SceneGpu {
    pub fn prepare(&mut self, world: &RenderWorld) -> Result<(), BoxError> {
        for item in &world.instances {
            let key = (
                item.material.emissive_texture,
                item.material.occlusion_texture,
            );
            if self.extras.contains_key(&key) {
                continue;
            }
            let emission = &self
                .textures
                .get(
                    &key.0
                        .unwrap_or(govno_render::TextureHandle::from_parts(0, 2)),
                )
                .ok_or("unknown emissive texture")?
                ._texture;
            let ao = &self
                .textures
                .get(
                    &key.1
                        .unwrap_or(govno_render::TextureHandle::from_parts(0, 1)),
                )
                .ok_or("unknown occlusion texture")?
                ._texture;
            let frames = self.cameras.len() as u32;
            let mut owner = ExtraSet {
                ctx: self.ctx.clone(),
                pool: vk::DescriptorPool::null(),
                sets: Vec::new(),
            };
            // SAFETY: new descriptors have no users; referenced texture handles stay
            // live until remove_texture quiesces GPU and clears this cache first.
            unsafe {
                let sizes = [
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(frames),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(frames * 2),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::SAMPLER)
                        .descriptor_count(frames * 2),
                ];
                owner.pool = self.ctx.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(frames)
                        .pool_sizes(&sizes),
                    None,
                )?;
                let layouts = vec![self.layout; frames as usize];
                owner.sets = self.ctx.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(owner.pool)
                        .set_layouts(&layouts),
                )?;
                for set in &owner.sets {
                    for (binding, texture) in [(1, emission), (3, ao)] {
                        let image = [vk::DescriptorImageInfo::default()
                            .image_view(texture.image.view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                        let sampler = [vk::DescriptorImageInfo::default().sampler(texture.sampler)];
                        self.ctx.device.update_descriptor_sets(
                            &[
                                vk::WriteDescriptorSet::default()
                                    .dst_set(*set)
                                    .dst_binding(binding)
                                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                    .image_info(&image),
                                vk::WriteDescriptorSet::default()
                                    .dst_set(*set)
                                    .dst_binding(binding + 1)
                                    .descriptor_type(vk::DescriptorType::SAMPLER)
                                    .image_info(&sampler),
                            ],
                            &[],
                        );
                    }
                }
            }
            self.extras.insert(key, owner);
        }
        Ok(())
    }
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
                emissive_texture: None,
                occlusion_texture: None,
                emissive: Vec3::new(0.25, 0.5, 0.75),
                occlusion_strength: 0.6,
                normal_texture: None,
                metallic_roughness_texture: None,
                normal_scale: 1.0,
                base_color_texture: None,
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
        assert_eq!(value(112), 0.25);
        assert_eq!(value(120), 0.75);
        assert_eq!(value(124), 0.6);
    }
}

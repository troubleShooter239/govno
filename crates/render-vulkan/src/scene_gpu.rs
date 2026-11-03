use crate::{
    context::DeviceContext,
    memory::{Buffer, Memory},
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{DrawInstance, RenderExtent, RenderWorld};
use gpu_allocator::MemoryLocation;
use std::{rc::Rc, sync::Arc};

pub(crate) const CAMERA_BYTES: u64 =
    112 + 32 * govno_render::MAX_POINT_LIGHTS as u64 + 80 + 144 + 16 + 288 + 96;
const OBJECT_PARAMETERS_BYTES: usize = 128;
pub(crate) const OBJECT_BYTES: u64 = 160;
pub(crate) struct SceneGpu {
    bindless: Option<crate::bindless::Bindless>,
    shadow: Option<crate::shadow::ShadowMap>,
    specular: crate::specular::SpecularGpu,
    loaded_probes: [bool; 3],
    ctx: Arc<DeviceContext>,
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    cameras: Vec<Buffer>,
    objects: Vec<Buffer>,
    object_capacities: Vec<usize>,
    object_counts: Vec<usize>,
    object_bytes: Vec<u8>,
    texture: Option<crate::texture::Texture>,
    extras: std::collections::HashMap<
        (
            Option<govno_render::TextureHandle>,
            Option<govno_render::TextureHandle>,
        ),
        ExtraSet,
    >,
    retired_extras: crate::mesh::RetirementQueue<ExtraSet>,
    memory: Rc<Memory>,
    textures: std::collections::HashMap<govno_render::TextureHandle, TextureSlot>,
}
impl SceneGpu {
    pub fn new(memory: Rc<Memory>, frames: u32) -> Result<Self, BoxError> {
        let mut owner = Self {
            bindless: if memory.ctx.bindless {
                Some(crate::bindless::Bindless::new(memory.ctx.clone(), frames)?)
            } else {
                None
            },
            shadow: None,
            specular: crate::specular::SpecularGpu::new(memory.clone())?,
            loaded_probes: [false, false, false],
            ctx: memory.ctx.clone(),
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            cameras: Vec::new(),
            objects: Vec::new(),
            object_capacities: Vec::new(),
            object_counts: vec![0; frames as usize],
            object_bytes: Vec::new(),
            texture: None,
            extras: Default::default(),
            retired_extras: Default::default(),
            memory: memory.clone(),
            textures: Default::default(),
        };
        for _ in 0..frames {
            owner.objects.push(Buffer::new(
                memory.clone(),
                OBJECT_BYTES * 64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
            owner.object_capacities.push(64);
            owner.cameras.push(Buffer::new(
                memory.clone(),
                CAMERA_BYTES,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
            )?);
        }
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
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
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(frames * 2),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(frames),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(frames * 5),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(frames * 3),
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
        owner.shadow = Some(crate::shadow::ShadowMap::new(memory.clone(), owner.layout)?);
        owner.shadow().bind_images(&owner.sets);
        for frame in 0..frames as usize {
            owner.bind_objects(frame);
        }
        owner.specular.bind(&owner.sets);
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
    pub fn set_environment_gpu(
        &mut self,
        data: &govno_render::EnvironmentRadiance,
        probe: u32,
    ) -> Result<(), BoxError> {
        self.specular.process(data, probe)?;
        self.loaded_probes[probe as usize] = true;
        Ok(())
    }
    pub fn set_specular_environment(
        &mut self,
        data: &govno_render::SpecularEnvironment,
    ) -> Result<(), BoxError> {
        self.specular.replace(data)?;
        self.loaded_probes[0] = true;
        self.specular.bind(&self.sets);
        for slot in self.textures.values() {
            self.specular.bind(&slot.sets);
        }
        Ok(())
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
        if let Some(table) = &mut self.bindless {
            table.replace_all(0, &texture);
            table.changed();
        }
        self.texture = Some(texture);
        Ok(())
    }
    /// Call only after waiting for this frame slot's submission timeline value.
    pub fn update(
        &mut self,
        frame: usize,
        world: &RenderWorld,
        extent: RenderExtent,
    ) -> Result<(), BoxError> {
        self.cameras[frame].write(&frame_constants(world, extent, self.loaded_probes[0])?)?;
        self.update_objects(frame, world)?;
        if let Some(table) = &mut self.bindless {
            table.update(
                frame,
                self.texture.as_ref().ok_or("missing default texture")?,
                self.textures
                    .values()
                    .map(|slot| (slot.index, &slot._texture)),
            );
        }
        Ok(())
    }
    fn bind_objects(&self, frame: usize) {
        let bind = |set| {
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(self.objects[frame].handle)
                .range(self.object_capacities[frame] as u64 * OBJECT_BYTES)];
            // SAFETY: only this completed frame slot is changed; the buffer remains
            // owned through its next submission, including when its capacity grows.
            unsafe {
                self.ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(10)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info)],
                    &[],
                );
            }
        };
        bind(self.sets[frame]);
        for slot in self.textures.values() {
            if let Some(&set) = slot.sets.get(frame) {
                bind(set);
            }
        }
    }
    fn update_objects(&mut self, frame: usize, world: &RenderWorld) -> Result<(), BoxError> {
        let limit = self.ctx.capabilities.max_storage_buffer_range as usize / OBJECT_BYTES as usize;
        let count = world.instances.len();
        if count > limit || count > u32::MAX as usize {
            return Err("GPU Scene exceeds storage-buffer/object-index limits".into());
        }
        if count > self.object_capacities[frame] {
            let capacity = count.next_power_of_two().min(limit);
            let buffer = Buffer::new(
                self.memory.clone(),
                capacity as u64 * OBJECT_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?;
            self.objects[frame] = buffer;
            self.object_capacities[frame] = capacity;
            self.bind_objects(frame);
        }
        self.object_bytes.clear();
        for instance in &world.instances {
            self.object_bytes
                .extend_from_slice(&encode_object_parameters(instance));
            // Texture indices are stable until quiesced removal; handles stay CPU-only.
            let m = &instance.material;
            let index =
                |handle: Option<govno_render::TextureHandle>, fallback| -> Result<u32, BoxError> {
                    match handle {
                        Some(h) => Ok(self
                            .textures
                            .get(&h)
                            .ok_or("stale GPU Scene texture")?
                            .index),
                        None => Ok(fallback),
                    }
                };
            let indices = if self.bindless.is_some() {
                [
                    index(m.base_color_texture, 0)?,
                    index(m.normal_texture, 1)?,
                    index(m.metallic_roughness_texture, 2)?,
                    index(m.emissive_texture, 3)?,
                    index(m.occlusion_texture, 2)?,
                    0,
                    0,
                    0,
                ]
            } else {
                [0; 8]
            };
            for index in indices {
                self.object_bytes.extend_from_slice(&index.to_ne_bytes());
            }
        }
        if count > 0 {
            self.objects[frame].write(&self.object_bytes)?;
        }
        self.object_counts[frame] = count;
        Ok(())
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"object_stride":OBJECT_BYTES,"frame_objects":self.object_counts,"frame_capacities":self.object_capacities,
            "bindless":self.bindless.as_ref().map(|t|t.report())})
    }
    pub fn shadow(&self) -> &crate::shadow::ShadowMap {
        self.shadow.as_ref().expect("shadow initialized")
    }
    pub fn texture_layout(&self) -> Option<vk::DescriptorSetLayout> {
        self.bindless.as_ref().map(|t| t.layout)
    }
    pub fn objects_binding(&self, frame: usize) -> crate::graph::Binding {
        crate::graph::Binding::Buffer {
            handle: self.objects[frame].handle,
            size: self.object_capacities[frame] as u64 * OBJECT_BYTES,
        }
    }
    pub fn global_sets(&self, frame: usize) -> Option<[vk::DescriptorSet; 2]> {
        self.bindless
            .as_ref()
            .map(|t| [self.sets[frame], t.sets[frame]])
    }
}
impl Drop for SceneGpu {
    fn drop(&mut self) {
        self.shadow.take();
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
fn encode_object_parameters(instance: &DrawInstance) -> [u8; OBJECT_PARAMETERS_BYTES] {
    let mut data = [0.0; 32];
    data[..16].copy_from_slice(&instance.transform.matrix().to_cols_array());
    let inverse_scale_squared = (instance.transform.scale * instance.transform.scale).recip();
    data[16..20].copy_from_slice(
        &inverse_scale_squared
            .extend(instance.material.roughness)
            .to_array(),
    );
    data[20..24].copy_from_slice(
        &instance
            .material
            .base_color
            .extend(instance.material.metallic)
            .to_array(),
    );
    data[24] = instance.material.normal_scale;
    data[25] = if instance.material.normal_texture.is_some() {
        1.0
    } else {
        0.0
    };
    data[28..32].copy_from_slice(
        &instance
            .material
            .emissive
            .extend(instance.material.occlusion_strength)
            .to_array(),
    );
    let mut bytes = [0; OBJECT_PARAMETERS_BYTES];
    for (destination, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(data) {
        destination.copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

struct TextureSlot {
    index: u32,
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
        if self.textures.contains_key(&handle) {
            return Err("duplicate GPU texture handle".into());
        }
        let texture = crate::texture::Texture::with_sampling(self.memory.clone(), data, sampling)?;
        let frames = self.cameras.len() as u32;
        let mut slot = TextureSlot {
            index: if let Some(table) = &mut self.bindless {
                table.allocate()?
            } else {
                0
            },
            ctx: self.ctx.clone(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            _texture: texture,
            color_space: data.color_space,
        };
        if self.bindless.is_some() {
            self.textures.insert(handle, slot);
            return Ok(());
        }
        // SAFETY: new descriptor sets have no GPU users; retained cameras/textures
        // outlive the slot. Partial allocation is owned by its pool.
        unsafe {
            let sizes = [
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::SAMPLER,
            ]
            .map(|ty| {
                vk::DescriptorPoolSize::default().ty(ty).descriptor_count(
                    if ty == vk::DescriptorType::STORAGE_BUFFER {
                        frames * 2
                    } else if ty == vk::DescriptorType::UNIFORM_BUFFER {
                        frames
                    } else if ty == vk::DescriptorType::SAMPLED_IMAGE {
                        frames * 5
                    } else {
                        frames * 3
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
            for (frame, (set, camera)) in slot.sets.iter().zip(&self.cameras).enumerate() {
                let objects = [vk::DescriptorBufferInfo::default()
                    .buffer(self.objects[frame].handle)
                    .range(self.object_capacities[frame] as u64 * OBJECT_BYTES)];
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
                            .dst_binding(10)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&objects),
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
        self.shadow().bind_images(&slot.sets);
        self.specular.bind(&slot.sets);
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
        self.validate_material(material)?;
        self.material_sets(frame, material)
    }
    pub fn validate_material(&self, material: &govno_render::Material) -> Result<(), BoxError> {
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
        Ok(())
    }
    fn material_sets(
        &self,
        frame: usize,
        material: &govno_render::Material,
    ) -> Result<[vk::DescriptorSet; 4], BoxError> {
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
        self.retired_extras.collect(u64::MAX);
        if let Some(table) = &mut self.bindless {
            let index = self
                .textures
                .get(&handle)
                .ok_or("unknown or removed texture handle")?
                .index;
            table.replace_all(
                index,
                self.texture.as_ref().ok_or("missing default texture")?,
            );
            table.release(index);
        }
        self.textures
            .remove(&handle)
            .ok_or("unknown or removed texture handle")?;
        Ok(())
    }
}

struct ExtraSet {
    used: bool,
    ctx: Arc<DeviceContext>,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}
impl Drop for ExtraSet {
    fn drop(&mut self) {
        // SAFETY: sets are released after their last submission completes, or after quiescence.
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
impl SceneGpu {
    pub fn collect_descriptors(&mut self, completed: u64) {
        self.retired_extras.collect(completed);
    }
    pub fn prepare(&mut self, world: &RenderWorld, submitted: u64) -> Result<(), BoxError> {
        if self.bindless.is_some() {
            return Ok(());
        }
        for entry in self.extras.values_mut() {
            entry.used = false;
        }
        let result = self.prepare_materials(world);
        // Retire unused pools even if preparing a later material failed.
        // Earlier single-queue submissions may still reference these sets.
        for (_, entry) in self.extras.extract_if(|_, entry| !entry.used) {
            self.retired_extras.retire(submitted, entry);
        }
        result
    }
    fn prepare_materials(&mut self, world: &RenderWorld) -> Result<(), BoxError> {
        for item in &world.instances {
            let key = (
                item.material.emissive_texture,
                item.material.occlusion_texture,
            );
            if let Some(entry) = self.extras.get_mut(&key) {
                entry.used = true;
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
                used: true,
                ctx: self.ctx.clone(),
                pool: vk::DescriptorPool::null(),
                sets: Vec::new(),
            };
            // SAFETY: new descriptors have no users; referenced texture handles stay
            // live until remove_texture quiesces GPU and clears this cache first.
            unsafe {
                let sizes = [
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(frames * 2),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(frames),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(frames * 5),
                    vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::SAMPLER)
                        .descriptor_count(frames * 3),
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

fn frame_constants(
    world: &RenderWorld,
    extent: RenderExtent,
    gpu_sh: bool,
) -> Result<Vec<u8>, BoxError> {
    let mut data = world
        .camera
        .view_projection(extent)?
        .to_cols_array()
        .to_vec();
    data.extend_from_slice(&world.camera.position.extend(1.0).to_array());
    data.extend_from_slice(&world.light_direction.normalize().extend(0.0).to_array());
    // Exact small integer count in the previously reserved fourth component.
    data.extend_from_slice(
        &world
            .light_radiance
            .extend(world.point_lights.len() as f32)
            .to_array(),
    );
    for light in &world.point_lights {
        data.extend_from_slice(&light.position.extend(light.range).to_array());
        data.extend_from_slice(&light.intensity.extend(0.0).to_array());
    }
    data.resize((112 + 32 * govno_render::MAX_POINT_LIGHTS) / 4, 0.0);
    let matrix = match world.directional_shadow {
        Some(shadow) => shadow.view_projection(world.light_direction)?,
        None => glam::Mat4::IDENTITY,
    };
    data.extend_from_slice(&matrix.to_cols_array());
    data.extend_from_slice(&[
        world.directional_shadow.map_or(0.0, |s| s.bias),
        if world.directional_shadow.is_some() {
            1.0
        } else {
            0.0
        },
        0.0,
        0.0,
    ]);
    let black = [glam::Vec3::ZERO; 9];
    let coefficients = world
        .diffuse_environment
        .as_ref()
        .map_or(&black, |environment| environment.coefficients());
    for coefficient in coefficients {
        data.extend_from_slice(&coefficient.extend(0.0).to_array());
    }
    data.extend_from_slice(&[
        if world.specular_environment { 1.0 } else { 0.0 },
        if world.diffuse_environment.is_some() {
            1.0
        } else {
            0.0
        },
        world.local_probes.len() as f32,
        if gpu_sh { 1.0 } else { 0.0 },
    ]);
    if let Some(shadow) = world.directional_shadow {
        let cascades = shadow.cascades(&world.camera, extent, world.light_direction)?;
        for matrix in cascades.matrices {
            data.extend_from_slice(&matrix.to_cols_array());
        }
        data.extend_from_slice(&cascades.splits);
        data.extend_from_slice(&[
            world.camera.near,
            shadow.cascade_blend,
            if shadow.debug_cascades { 1.0 } else { 0.0 },
            0.0,
        ]);
    } else {
        data.resize(data.len() + 72, 0.0);
    }
    for i in 0..2 {
        if let Some(probe) = world.local_probes.get(i) {
            data.extend_from_slice(&probe.position.extend(probe.slot as f32).to_array());
            data.extend_from_slice(&probe.bounds_min.extend(probe.fade).to_array());
            data.extend_from_slice(&probe.bounds_max.extend(probe.strength).to_array());
        } else {
            data.resize(data.len() + 12, 0.0);
        }
    }
    Ok(floats(&data))
}

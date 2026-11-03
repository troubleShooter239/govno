use crate::{context::DeviceContext, texture::Texture};
use ash::vk;
use govno_core::BoxError;
use std::sync::Arc;
/// One texture/sampler table per frame, updated only after its timeline wait.
pub(crate) struct Bindless {
    ctx: Arc<DeviceContext>,
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    capacity: u32,
    next: u32,
    free: Vec<u32>,
    revision: u64,
    frame_revisions: Vec<u64>,
}
impl Bindless {
    pub fn new(ctx: Arc<DeviceContext>, frames: u32) -> Result<Self, BoxError> {
        let capacity = ctx.capabilities.bindless_capacity;
        let mut owner = Self {
            ctx,
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            sets: vec![],
            capacity,
            next: 1,
            free: vec![],
            revision: 1,
            frame_revisions: vec![0; frames as usize],
        };
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(capacity)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(capacity)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND; 2];
        let mut indexing =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        // SAFETY: features/limits are checked at device creation. Handles immediately
        // gain an owner, including on partial construction failure.
        unsafe {
            owner.layout = owner.ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .push_next(&mut indexing),
                None,
            )?;
            let sizes = [
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::SAMPLER,
            ]
            .map(|ty| {
                vk::DescriptorPoolSize::default()
                    .ty(ty)
                    .descriptor_count(capacity * frames)
            });
            owner.pool = owner.ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(frames)
                    .pool_sizes(&sizes),
                None,
            )?;
            owner.sets = owner.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(owner.pool)
                    .set_layouts(&vec![owner.layout; frames as usize]),
            )?;
        }
        Ok(owner)
    }
    pub fn allocate(&mut self) -> Result<u32, BoxError> {
        let index = if let Some(i) = self.free.pop() {
            i
        } else {
            if self.next >= self.capacity {
                return Err("bindless texture capacity exceeded".into());
            }
            let i = self.next;
            self.next += 1;
            i
        };
        self.changed();
        Ok(index)
    }
    pub fn release(&mut self, index: u32) {
        self.free.push(index);
        self.changed();
    }
    pub fn changed(&mut self) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("descriptor revision exhausted");
    }
    fn write(&self, set: vk::DescriptorSet, index: u32, texture: &Texture) {
        let image = [vk::DescriptorImageInfo::default()
            .image_view(texture.image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let sampler = [vk::DescriptorImageInfo::default().sampler(texture.sampler)];
        // SAFETY: caller completed this set's GPU users; SceneGpu retains the texture.
        unsafe {
            self.ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .dst_array_element(index)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&image),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .dst_array_element(index)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&sampler),
                ],
                &[],
            );
        }
    }
    /// Quiesced replacement/removal, before destroying the previous texture.
    pub fn replace_all(&self, index: u32, texture: &Texture) {
        for &set in &self.sets {
            self.write(set, index, texture);
        }
    }
    pub fn update<'a>(
        &mut self,
        frame: usize,
        shared: &Texture,
        textures: impl Iterator<Item = (u32, &'a Texture)>,
    ) {
        if self.frame_revisions[frame] == self.revision {
            return;
        }
        let set = self.sets[frame];
        self.write(set, 0, shared);
        for (index, texture) in textures {
            self.write(set, index, texture);
        }
        self.frame_revisions[frame] = self.revision;
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"capacity":self.capacity,"live_textures":self.next-self.free.len() as u32})
    }
}
impl Drop for Bindless {
    fn drop(&mut self) {
        // SAFETY: SceneGpu destruction follows GPU completion and pipeline destruction.
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

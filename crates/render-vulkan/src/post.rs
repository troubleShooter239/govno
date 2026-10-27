use crate::{
    context::DeviceContext,
    memory::{ImageResource, Memory},
    pipeline::ForwardPipeline,
};
use ash::vk;
use govno_core::BoxError;
use std::{rc::Rc, sync::Arc};

pub(crate) const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub(crate) struct PostProcess {
    ctx: Arc<DeviceContext>,
    pub targets: Vec<ImageResource>,
    pub effects: Option<crate::effects::Effects>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
    pub pipeline: Option<ForwardPipeline>,
}
impl PostProcess {
    pub fn new(
        memory: Rc<Memory>,
        extent: vk::Extent2D,
        output_format: vk::Format,
    ) -> Result<Self, BoxError> {
        // SAFETY: physical device belongs to the retained instance.
        let features = unsafe {
            memory
                .ctx
                .instance()
                .get_physical_device_format_properties(memory.ctx.physical, HDR_FORMAT)
        }
        .optimal_tiling_features;
        if !features.contains(
            vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::SAMPLED_IMAGE,
        ) {
            return Err("RGBA16F color attachment/sampling support is required for HDR".into());
        }
        let mut owner = Self {
            ctx: memory.ctx.clone(),
            targets: Vec::new(),
            effects: None,
            layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            pipeline: None,
        };
        for _ in 0..2 {
            owner
                .targets
                .push(ImageResource::hdr(memory.clone(), extent)?);
        }
        owner.effects = Some(crate::effects::Effects::new(
            memory.clone(),
            extent,
            &owner.targets,
        )?);
        // SAFETY: all successful handles are immediately owned. Descriptors are
        // initialized before use and remain immutable for the swapchain lifetime.
        unsafe {
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            ];
            owner.layout = owner.ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(4),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(2),
            ];
            owner.pool = owner.ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2)
                    .pool_sizes(&sizes),
                None,
            )?;
            owner.sets = owner.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(owner.pool)
                    .set_layouts(&[owner.layout; 2]),
            )?;
            for (i, (set, target)) in owner.sets.iter().zip(&owner.targets).enumerate() {
                let images = [vk::DescriptorImageInfo::default()
                    .image_view(target.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let effects = owner.effects.as_ref().ok_or("post effects missing")?;
                let bloom = [vk::DescriptorImageInfo::default()
                    .image_view(effects.bloom[i].view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let state = [vk::DescriptorBufferInfo::default()
                    .buffer(effects.exposure.handle)
                    .range(16)];
                owner.ctx.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&bloom),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&state),
                    ],
                    &[],
                );
                owner.ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&images)],
                    &[],
                );
            }
        }
        owner.pipeline = Some(ForwardPipeline::post(
            owner.ctx.clone(),
            output_format,
            owner.layout,
        )?);
        tracing::info!(width=extent.width,height=extent.height,format=?HDR_FORMAT,"per-frame HDR targets created");
        Ok(owner)
    }
}
impl Drop for PostProcess {
    fn drop(&mut self) {
        // SAFETY: swapchain quiesces GPU work before destroying post resources.
        // On partial construction failure, no commands reference these objects.
        self.pipeline.take();
        // SAFETY: users are quiesced as described above; handles belong to ctx.
        unsafe {
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

// Shader ABI: uint manualSRGB at byte 0, float linear exposure at byte 4.
pub(crate) const OUTPUT_BYTES: u32 = 8;
pub(crate) fn output_constants(
    settings: govno_render::PostProcessSettings,
    srgb_attachment: bool,
) -> [u8; OUTPUT_BYTES as usize] {
    let mut bytes = [0; OUTPUT_BYTES as usize];
    bytes[..4].copy_from_slice(&u32::from(!srgb_attachment).to_ne_bytes());
    bytes[4..].copy_from_slice(&settings.bloom.intensity.to_ne_bytes());
    bytes
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_abi_preserves_exposure_and_encoding_path() {
        for srgb in [false, true] {
            let bytes = output_constants(
                govno_render::PostProcessSettings {
                    exposure_ev: -1.0,
                    ..Default::default()
                },
                srgb,
            );
            assert_eq!(
                u32::from_ne_bytes(bytes[..4].try_into().unwrap()),
                u32::from(!srgb)
            );
            assert_eq!(f32::from_ne_bytes(bytes[4..].try_into().unwrap()), 0.08);
        }
    }
}

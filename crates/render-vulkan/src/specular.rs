use crate::{
    memory::{Buffer, ImageResource, Memory},
    mesh::Upload,
    swapchain::color_range,
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{
    BRDF_LUT_SIZE, SPECULAR_HEIGHT, SPECULAR_LEVELS, SPECULAR_WIDTH, SpecularEnvironment,
};
use gpu_allocator::MemoryLocation;
use std::rc::Rc;
pub(crate) struct SpecularGpu {
    memory: Rc<Memory>,
    pub radiance: ImageResource,
    lut: ImageResource,
    sampler: vk::Sampler,
}
impl SpecularGpu {
    pub fn new(memory: Rc<Memory>) -> Result<Self, BoxError> {
        let radiance = upload(
            memory.clone(),
            SPECULAR_WIDTH,
            SPECULAR_HEIGHT * SPECULAR_LEVELS,
            SpecularEnvironment::constant(glam::Vec3::ZERO)?.atlas(),
        )?;
        let lut = upload(
            memory.clone(),
            BRDF_LUT_SIZE,
            BRDF_LUT_SIZE,
            &govno_render::specular_brdf_lut(),
        )?;
        let mut owner = Self {
            memory,
            radiance,
            lut,
            sampler: vk::Sampler::null(),
        };
        // SAFETY: sampler belongs to the retained device; all image data is initialized.
        owner.sampler = unsafe {
            owner.memory.ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .max_lod(0.0)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        Ok(owner)
    }
    /// Caller quiesces GPU descriptor users before replacement.
    pub fn replace(&mut self, data: &SpecularEnvironment) -> Result<(), BoxError> {
        self.radiance = upload(
            self.memory.clone(),
            SPECULAR_WIDTH,
            SPECULAR_HEIGHT * SPECULAR_LEVELS,
            data.atlas(),
        )?;
        Ok(())
    }
    pub fn bind(&self, sets: &[vk::DescriptorSet]) {
        for set in sets {
            let radiance = [vk::DescriptorImageInfo::default()
                .image_view(self.radiance.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let lut = [vk::DescriptorImageInfo::default()
                .image_view(self.lut.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let sampler = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
            // SAFETY: caller supplies new or quiesced sets with bindings 6/7/8.
            // Images/sampler stay live until quiesced replacement or renderer teardown.
            unsafe {
                self.memory.ctx.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(6)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&radiance),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(7)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(8)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&lut),
                    ],
                    &[],
                );
            }
        }
    }
}
impl Drop for SpecularGpu {
    fn drop(&mut self) {
        // SAFETY: owner quiesces submissions before dropping resources.
        unsafe {
            self.memory.ctx.device.destroy_sampler(self.sampler, None);
        }
    }
}
fn upload(
    memory: Rc<Memory>,
    width: u32,
    height: u32,
    pixels: &[[f32; 4]],
) -> Result<ImageResource, BoxError> {
    let format = vk::Format::R32G32B32A32_SFLOAT;
    // SAFETY: physical device belongs to the retained instance.
    let features = unsafe {
        memory
            .ctx
            .instance()
            .get_physical_device_format_properties(memory.ctx.physical, format)
    }
    .optimal_tiling_features;
    if !features.contains(
        vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::TRANSFER_SRC,
    ) {
        return Err(
            "RGBA32F sampled/linear-filter/transfer support required for specular IBL".into(),
        );
    }
    let image = ImageResource::sampled(memory.clone(), vk::Extent2D { width, height }, 1, format)?;
    let bytes: Vec<u8> = pixels
        .iter()
        .flatten()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let mut staging = Buffer::new(
        memory.clone(),
        bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    )?;
    staging.write(&bytes)?;
    let ctx = memory.ctx.clone();
    Upload::run(ctx.clone(), |cmd| {
        // SAFETY: upload helper retains command/fence; image/staging are live until
        // completion. CPU cooker supplies exact dimensions and finite RGBA pixels.
        unsafe {
            let before = [vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image.image)
                .subresource_range(color_range())
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
            ctx.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&before),
            );
            let region = [vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_buffer_to_image(
                cmd,
                staging.handle,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );
            let after = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image.image)
                .subresource_range(color_range())
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
            ctx.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&after),
            );
        }
    })?;
    Ok(image)
}

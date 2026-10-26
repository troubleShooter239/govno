use crate::{
    memory::{Buffer, ImageResource, Memory},
    mesh::Upload,
    swapchain::color_range,
};
use ash::vk;
use govno_core::BoxError;
use govno_render::TextureData;
use gpu_allocator::MemoryLocation;
use std::rc::Rc;

pub(crate) struct Texture {
    memory: Rc<Memory>,
    pub image: ImageResource,
    pub sampler: vk::Sampler,
}
impl Texture {
    pub fn new(memory: Rc<Memory>, data: &TextureData) -> Result<Self, BoxError> {
        data.validate()?;
        // SAFETY: querying capabilities for the retained physical device.
        let props = unsafe {
            memory
                .ctx
                .instance()
                .get_physical_device_properties(memory.ctx.physical)
        };
        if data.width > props.limits.max_image_dimension2_d
            || data.height > props.limits.max_image_dimension2_d
        {
            return Err("texture dimensions exceed GPU limits".into());
        }
        let image = ImageResource::sampled(
            memory.clone(),
            vk::Extent2D {
                width: data.width,
                height: data.height,
            },
        )?;
        let mut owner = Self {
            memory: memory.clone(),
            image,
            sampler: vk::Sampler::null(),
        };
        let mut staging = Buffer::new(
            memory.clone(),
            data.rgba8.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging.write(&data.rgba8)?;
        let ctx = memory.ctx.clone();
        Upload::run(ctx.clone(), |cmd| {
            // SAFETY: image/staging remain alive through synchronous completion.
            // Image is new, ranges and copy sizes match its validated dimensions.
            unsafe {
                let before = [vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .image(owner.image.image)
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
                        width: data.width,
                        height: data.height,
                        depth: 1,
                    })];
                ctx.device.cmd_copy_buffer_to_image(
                    cmd,
                    staging.handle,
                    owner.image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &region,
                );
                let after = [vk::ImageMemoryBarrier2::default()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .image(owner.image.image)
                    .subresource_range(color_range())
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
                ctx.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&after),
                );
            }
        })?;
        // SAFETY: ordinary linear repeat sampler, one mip, no optional features.
        owner.sampler = unsafe {
            ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .max_lod(0.0),
                None,
            )?
        };
        Ok(owner)
    }
}
impl Drop for Texture {
    fn drop(&mut self) {
        // SAFETY: descriptor owner/renderer quiesces referring submissions first.
        unsafe {
            self.memory.ctx.device.destroy_sampler(self.sampler, None);
        }
    }
}

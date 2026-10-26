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
        Self::with_sampling(memory, data, Default::default())
    }
    pub fn with_sampling(
        memory: Rc<Memory>,
        data: &TextureData,
        sampling: govno_render::TextureSampling,
    ) -> Result<Self, BoxError> {
        data.validate()?;
        let format = if data.color_space == govno_render::TextureColorSpace::Srgb {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        let levels = sampling.mip_levels(data.width, data.height);
        // SAFETY: physical device belongs to the retained instance.
        let features = unsafe {
            memory
                .ctx
                .instance()
                .get_physical_device_format_properties(memory.ctx.physical, format)
        }
        .optimal_tiling_features;
        if levels > 1
            && !features.contains(
                vk::FormatFeatureFlags::BLIT_SRC
                    | vk::FormatFeatureFlags::BLIT_DST
                    | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
            )
        {
            return Err("texture format linear blit unsupported: cannot generate mipmaps".into());
        }
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
            levels,
            format,
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
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .image(owner.image.image)
                    .subresource_range(color_range().level_count(levels))
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
                for level in 1..levels {
                    let ready = [vk::ImageMemoryBarrier2::default()
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .image(owner.image.image)
                        .subresource_range(color_range().base_mip_level(level - 1))
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
                    ctx.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&ready),
                    );
                    let size = |mip: u32| vk::Offset3D {
                        x: (data.width >> mip).max(1) as i32,
                        y: (data.height >> mip).max(1) as i32,
                        z: 1,
                    };
                    let layers = |mip| {
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(mip)
                            .layer_count(1)
                    };
                    let blit = [vk::ImageBlit::default()
                        .src_subresource(layers(level - 1))
                        .src_offsets([vk::Offset3D::default(), size(level - 1)])
                        .dst_subresource(layers(level))
                        .dst_offsets([vk::Offset3D::default(), size(level)])];
                    ctx.device.cmd_blit_image(
                        cmd,
                        owner.image.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        owner.image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &blit,
                        vk::Filter::LINEAR,
                    );
                }
                let after: Vec<_> = (0..levels)
                    .map(|level| {
                        vk::ImageMemoryBarrier2::default()
                            .old_layout(if level + 1 == levels {
                                vk::ImageLayout::TRANSFER_DST_OPTIMAL
                            } else {
                                vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                            })
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                            .src_access_mask(
                                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
                            )
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .image(owner.image.image)
                            .subresource_range(color_range().base_mip_level(level))
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    })
                    .collect();
                ctx.device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&after),
                );
            }
        })?;
        // SAFETY: validated mip count and core sampler modes; no optional features.
        owner.sampler = unsafe {
            ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(filter(sampling.mag))
                    .min_filter(filter(sampling.min))
                    .mipmap_mode(if sampling.mip == govno_render::MipFilter::Linear {
                        vk::SamplerMipmapMode::LINEAR
                    } else {
                        vk::SamplerMipmapMode::NEAREST
                    })
                    .address_mode_u(wrap(sampling.wrap_u))
                    .address_mode_v(wrap(sampling.wrap_v))
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .max_lod((levels - 1) as f32),
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

fn filter(value: govno_render::TextureFilter) -> vk::Filter {
    match value {
        govno_render::TextureFilter::Nearest => vk::Filter::NEAREST,
        govno_render::TextureFilter::Linear => vk::Filter::LINEAR,
    }
}
fn wrap(value: govno_render::TextureWrap) -> vk::SamplerAddressMode {
    match value {
        govno_render::TextureWrap::Repeat => vk::SamplerAddressMode::REPEAT,
        govno_render::TextureWrap::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
        govno_render::TextureWrap::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
    }
}

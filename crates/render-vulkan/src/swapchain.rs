use crate::context::DeviceContext;
use crate::pipeline::ForwardPipeline;
use ash::vk;
use govno_core::BoxError;
use govno_render::RenderExtent;
use std::sync::Arc;

pub(crate) fn choose_extent(
    caps: &vk::SurfaceCapabilitiesKHR,
    requested: RenderExtent,
) -> Option<vk::Extent2D> {
    if requested.is_zero() {
        return None;
    }
    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: requested
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: requested
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    };
    (extent.width > 0 && extent.height > 0).then_some(extent)
}
pub(crate) fn image_count(caps: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let desired = caps.min_image_count.saturating_add(1);
    if caps.max_image_count == 0 {
        desired
    } else {
        desired.min(caps.max_image_count)
    }
}
pub(crate) struct Swapchain {
    pub ctx: Arc<DeviceContext>,
    pub handle: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub finished: Vec<vk::Semaphore>,
    pub present_fences: Vec<vk::Fence>,
    pub pending_present: Vec<bool>,
    pub pipeline: ForwardPipeline,
    pub depth: Vec<crate::memory::DepthImage>,
    pub srgb_attachment: bool,
}
impl Swapchain {
    pub fn new(
        memory: std::rc::Rc<crate::memory::Memory>,
        requested: RenderExtent,
        old: vk::SwapchainKHR,
        scene_layout: vk::DescriptorSetLayout,
    ) -> Result<Option<Self>, BoxError> {
        let ctx = memory.ctx.clone();
        let depth_format = ctx.depth_format()?;
        let caps = ctx.capabilities()?;
        let Some(extent) = choose_extent(&caps, requested) else {
            return Ok(None);
        };
        let formats = ctx.formats()?;
        let format = select_format(&formats)?;
        let alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .into_iter()
        .find(|f| caps.supported_composite_alpha.contains(*f))
        .ok_or("no composite alpha mode")?;
        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(ctx.surface())
            .min_image_count(image_count(&caps))
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old);
        let pipeline =
            ForwardPipeline::new(ctx.clone(), format.format, depth_format, scene_layout)?;
        // SAFETY: device/surface are live, the caller quiesces the old swapchain.
        // Each child handle is recorded in its RAII owner immediately after creation.
        unsafe {
            let handle = ctx.swapchain.create_swapchain(&info, None)?;
            let mut chain = Self {
                ctx,
                handle,
                extent,
                images: Vec::new(),
                views: Vec::new(),
                finished: Vec::new(),
                present_fences: Vec::new(),
                pending_present: Vec::new(),
                pipeline,
                depth: Vec::new(),
                srgb_attachment: matches!(
                    format.format,
                    vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                ),
            };
            for _ in 0..2 {
                chain.depth.push(crate::memory::DepthImage::new(
                    memory.clone(),
                    extent,
                    depth_format,
                )?);
            }
            chain.images = chain.ctx.swapchain.get_swapchain_images(handle)?;
            for image in &chain.images {
                let view = vk::ImageViewCreateInfo::default()
                    .image(*image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(color_range());
                chain
                    .views
                    .push(chain.ctx.device.create_image_view(&view, None)?);
                chain.finished.push(
                    chain
                        .ctx
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
                );
                if chain.ctx.present_fences {
                    chain.present_fences.push(
                        chain
                            .ctx
                            .device
                            .create_fence(&vk::FenceCreateInfo::default(), None)?,
                    );
                }
                chain.pending_present.push(false);
            }
            tracing::info!(
                width = extent.width,
                height = extent.height,
                images = chain.images.len(),
                ?format,
                "swapchain created"
            );
            Ok(Some(chain))
        }
    }
    pub fn quiesce(&mut self) -> Result<(), BoxError> {
        self.ctx.wait_idle()?;
        // SAFETY: only fences attached to queued presents are waited; all handles
        // belong to this device and no concurrent host operations are possible.
        unsafe {
            for (index, pending) in self.pending_present.iter_mut().enumerate() {
                if *pending {
                    self.ctx.device.wait_for_fences(
                        &[self.present_fences[index]],
                        true,
                        u64::MAX,
                    )?;
                    *pending = false;
                }
            }
        }
        Ok(())
    }
}
impl Drop for Swapchain {
    fn drop(&mut self) {
        if let Err(error) = self.quiesce() {
            tracing::error!(%error, "swapchain cleanup wait failed");
        }
        // SAFETY: submissions and, when supported, presentation fences are complete.
        // The documented baseline WSI fallback uses device-idle. On DEVICE_LOST,
        // Vulkan allows destruction of device children.
        unsafe {
            for fence in &self.present_fences {
                self.ctx.device.destroy_fence(*fence, None);
            }
            for semaphore in &self.finished {
                self.ctx.device.destroy_semaphore(*semaphore, None);
            }
            for view in &self.views {
                self.ctx.device.destroy_image_view(*view, None);
            }
            self.ctx.swapchain.destroy_swapchain(self.handle, None);
        }
    }
}
pub(crate) fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn caps() -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 0,
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 64,
                height: 64,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            ..Default::default()
        }
    }
    #[test]
    fn minimized_extent_is_not_clamped_up() {
        assert!(
            choose_extent(
                &caps(),
                RenderExtent {
                    width: 0,
                    height: 720
                }
            )
            .is_none()
        );
        assert!(
            choose_extent(
                &caps(),
                RenderExtent {
                    width: 1280,
                    height: 0
                }
            )
            .is_none()
        );
    }
    #[test]
    fn variable_extent_clamps_to_surface_limits() {
        let size = choose_extent(
            &caps(),
            RenderExtent {
                width: 9000,
                height: 10,
            },
        )
        .unwrap();
        assert_eq!((size.width, size.height), (4096, 64));
    }
    #[test]
    fn fixed_surface_extent_wins() {
        let mut caps = caps();
        caps.current_extent = vk::Extent2D {
            width: 800,
            height: 600,
        };
        let size = choose_extent(
            &caps,
            RenderExtent {
                width: 1280,
                height: 720,
            },
        )
        .unwrap();
        assert_eq!((size.width, size.height), (800, 600));
        caps.current_extent.width = 0;
        assert!(
            choose_extent(
                &caps,
                RenderExtent {
                    width: 1280,
                    height: 720
                }
            )
            .is_none()
        );
    }
    #[test]
    fn image_limit_zero_means_unbounded() {
        let mut caps = caps();
        assert_eq!(image_count(&caps), 3);
        caps.max_image_count = 2;
        assert_eq!(image_count(&caps), 2);
    }
}

fn select_format(formats: &[vk::SurfaceFormatKHR]) -> Result<vk::SurfaceFormatKHR, BoxError> {
    for desired in [
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_SRGB,
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8B8A8_UNORM,
    ] {
        if let Some(format) = formats.iter().find(|f| {
            f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                && (f.format == desired || f.format == vk::Format::UNDEFINED)
        }) {
            return Ok(vk::SurfaceFormatKHR {
                format: desired,
                color_space: format.color_space,
            });
        }
    }
    Err("no supported SDR sRGB surface format; HDR output is not implemented".into())
}
#[cfg(test)]
mod format_tests {
    use super::*;
    #[test]
    fn srgb_attachment_is_preferred_over_manual_encoding() {
        let formats = [
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_SRGB,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        ];
        assert_eq!(
            select_format(&formats).unwrap().format,
            vk::Format::B8G8R8A8_SRGB
        );
        assert_eq!(
            select_format(&formats[..1]).unwrap().format,
            vk::Format::R8G8B8A8_UNORM
        );
    }
    #[test]
    fn unsupported_color_space_is_not_silently_treated_as_srgb() {
        assert!(
            select_format(&[vk::SurfaceFormatKHR {
                format: vk::Format::A2B10G10R10_UNORM_PACK32,
                color_space: vk::ColorSpaceKHR::HDR10_ST2084_EXT
            }])
            .is_err()
        );
    }
}

//! Bounded cache of completed frame targets. No image is shared by live frame slots.
use crate::memory::{ImageDescriptor, ImageResource, Memory, depth_range};
use ash::vk;
use govno_core::BoxError;
use std::{collections::VecDeque, rc::Rc};

const MAX_CACHED_BYTES: u64 = 128 * 1024 * 1024;
pub(crate) struct ImagePool {
    memory: Rc<Memory>,
    available: VecDeque<ImageResource>,
    cached_bytes: u64,
    created: u64,
    reused: u64,
}
impl ImagePool {
    pub fn new(memory: Rc<Memory>) -> Self {
        Self {
            memory,
            available: VecDeque::new(),
            cached_bytes: 0,
            created: 0,
            reused: 0,
        }
    }
    fn acquire(&mut self, descriptor: ImageDescriptor) -> Result<ImageResource, BoxError> {
        if let Some(i) = self
            .available
            .iter()
            .position(|image| image.descriptor == descriptor)
        {
            let image = self.available.remove(i).expect("matched pooled image");
            self.cached_bytes -= image.allocated_bytes();
            self.reused += 1;
            return Ok(image);
        }
        let image = ImageResource::with_usage(
            self.memory.clone(),
            descriptor.extent,
            descriptor.format,
            descriptor.usage,
            descriptor.range,
        )?;
        self.created += 1;
        Ok(image)
    }
    pub fn hdr(&mut self, extent: vk::Extent2D) -> Result<ImageResource, BoxError> {
        self.acquire(ImageDescriptor {
            extent,
            format: crate::post::HDR_FORMAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            range: crate::swapchain::color_range(),
        })
    }
    pub fn depth(
        &mut self,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<ImageResource, BoxError> {
        self.acquire(ImageDescriptor {
            extent,
            format,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            range: depth_range(),
        })
    }
    pub fn bloom(&mut self, extent: vk::Extent2D) -> Result<ImageResource, BoxError> {
        self.acquire(ImageDescriptor {
            extent,
            format: crate::post::HDR_FORMAT,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            range: crate::swapchain::color_range(),
        })
    }
    /// Caller has completed all queue users. Cached layout is intentionally discarded
    /// by the next graph's first write; descriptors are rebuilt before reuse.
    pub fn recycle(&mut self, image: ImageResource) {
        self.cached_bytes += image.allocated_bytes();
        self.available.push_back(image);
    }
    /// Trim after acquiring the replacement targets, including on rebuild failure.
    /// Otherwise a large retired size could evict the requested cached size first.
    /// Recycling transfers ownership only and does not allocate additional memory.
    pub fn trim(&mut self) {
        while self.cached_bytes > MAX_CACHED_BYTES {
            if let Some(old) = self.available.pop_front() {
                self.cached_bytes -= old.allocated_bytes();
            } else {
                break;
            }
        }
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"created_images":self.created,"reused_images":self.reused,
            "cached_images":self.available.len(),"cached_bytes":self.cached_bytes,"cache_limit_bytes":MAX_CACHED_BYTES})
    }
    pub fn clear(&mut self) {
        self.available.clear();
        self.cached_bytes = 0;
    }
}

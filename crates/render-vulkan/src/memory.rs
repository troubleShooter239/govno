use crate::context::DeviceContext;
use ash::vk;
use govno_core::BoxError;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
};
use std::{cell::RefCell, rc::Rc, sync::Arc};

pub(crate) struct Memory {
    // Allocator must be dropped before the retained device.
    allocator: RefCell<Allocator>,
    pub ctx: Arc<DeviceContext>,
    atom_size: u64,
    live_bytes: std::cell::Cell<u64>,
    device_local_bytes: std::cell::Cell<u64>,
    allocations: std::cell::Cell<u64>,
    total_allocations: std::cell::Cell<u64>,
    peak_device_local_bytes: std::cell::Cell<u64>,
}
impl Memory {
    pub fn new(ctx: Arc<DeviceContext>) -> Result<Rc<Self>, BoxError> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance().clone(),
            device: ctx.device.clone(),
            physical_device: ctx.physical,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        // SAFETY: valid physical device in retained instance.
        let atom_size = unsafe { ctx.instance().get_physical_device_properties(ctx.physical) }
            .limits
            .non_coherent_atom_size;
        Ok(Rc::new(Self {
            allocator: RefCell::new(allocator),
            ctx,
            atom_size,
            live_bytes: Default::default(),
            device_local_bytes: Default::default(),
            allocations: Default::default(),
            total_allocations: Default::default(),
            peak_device_local_bytes: Default::default(),
        }))
    }
    fn track(&self, a: &Allocation) {
        self.live_bytes.set(self.live_bytes.get() + a.size());
        self.allocations.set(self.allocations.get() + 1);
        self.total_allocations.set(self.total_allocations.get() + 1);
        if a.memory_properties()
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        {
            self.device_local_bytes
                .set(self.device_local_bytes.get() + a.size());
            self.peak_device_local_bytes.set(
                self.peak_device_local_bytes
                    .get()
                    .max(self.device_local_bytes.get()),
            );
        }
    }
    pub fn report(&self) -> serde_json::Value {
        let mut driver_usage = None;
        let mut driver_budget = None;
        if self.ctx.capabilities.memory_budget {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut properties =
                vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            // SAFETY: memory-budget extension was enabled and physical device is retained.
            unsafe {
                self.ctx
                    .instance()
                    .get_physical_device_memory_properties2(self.ctx.physical, &mut properties);
            }
            let heaps = &properties.memory_properties.memory_heaps
                [..properties.memory_properties.memory_heap_count as usize];
            let indices = heaps
                .iter()
                .enumerate()
                .filter_map(|(i, h)| {
                    h.flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
                        .then_some(i)
                })
                .collect::<Vec<_>>();
            driver_usage = Some(indices.iter().map(|&i| budget.heap_usage[i]).sum::<u64>());
            driver_budget = Some(indices.iter().map(|&i| budget.heap_budget[i]).sum::<u64>());
        }
        serde_json::json!({"driver_device_local_usage_bytes":driver_usage,"driver_device_local_budget_bytes":driver_budget,"live_allocations":self.allocations.get(),"total_resource_allocations":self.total_allocations.get(),"live_bytes":self.live_bytes.get(),"allocator_capacity_bytes":self.allocator.borrow().capacity(),"device_local_live_bytes":self.device_local_bytes.get(),"device_local_peak_bytes":self.peak_device_local_bytes.get()})
    }
    fn free(&self, allocation: Allocation) {
        let size = allocation.size();
        let local = allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        if let Err(error) = self.allocator.borrow_mut().free(allocation) {
            tracing::error!(%error, "GPU allocation cleanup failed");
        } else {
            self.live_bytes.set(self.live_bytes.get() - size);
            self.allocations.set(self.allocations.get() - 1);
            if local {
                self.device_local_bytes
                    .set(self.device_local_bytes.get() - size);
            }
        }
    }
}
fn aligned_size(size: u64, alignment: u64) -> u64 {
    size.div_ceil(alignment) * alignment
}

pub(crate) struct Buffer {
    memory: Rc<Memory>,
    pub handle: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
}
impl Buffer {
    pub fn new(
        memory: Rc<Memory>,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self, BoxError> {
        if size == 0 {
            return Err("zero-sized GPU buffer".into());
        }
        let mut buffer = Self {
            memory,
            handle: vk::Buffer::null(),
            allocation: None,
            size,
        };
        // SAFETY: device is live. Store handles/allocation in the owner before
        // further fallible calls. Allocation is freed only after destroying buffer.
        unsafe {
            buffer.handle = buffer.memory.ctx.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let mut requirements = buffer
                .memory
                .ctx
                .device
                .get_buffer_memory_requirements(buffer.handle);
            if location != MemoryLocation::GpuOnly {
                requirements.alignment = requirements.alignment.max(buffer.memory.atom_size);
                requirements.size = aligned_size(requirements.size, buffer.memory.atom_size);
            }
            let allocation =
                buffer
                    .memory
                    .allocator
                    .borrow_mut()
                    .allocate(&AllocationCreateDesc {
                        name: "buffer",
                        requirements,
                        location,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })?;
            buffer.memory.track(&allocation);
            buffer.allocation = Some(allocation);
            let allocation = buffer.allocation.as_ref().expect("allocated buffer");
            buffer.memory.ctx.device.bind_buffer_memory(
                buffer.handle,
                allocation.memory(),
                allocation.offset(),
            )?;
        }
        Ok(buffer)
    }
    /// Read mapped data after a GPU->HOST barrier and completion wait.
    pub fn read(&mut self) -> Result<Vec<u8>, BoxError> {
        let allocation = self.allocation.as_mut().ok_or("unallocated buffer")?;
        if !allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            // SAFETY: allocation is atom-aligned/padded; caller has waited for GPU writers.
            unsafe {
                self.memory.ctx.device.invalidate_mapped_memory_ranges(&[
                    vk::MappedMemoryRange::default()
                        .memory(allocation.memory())
                        .offset(allocation.offset())
                        .size(aligned_size(self.size, self.memory.atom_size)),
                ])?;
            }
        }
        Ok(allocation
            .mapped_slice_mut()
            .ok_or("buffer is not mapped")?[..self.size as usize]
            .to_vec())
    }
    /// Caller must ensure this range is no longer being read by the GPU.
    pub fn write(&mut self, bytes: &[u8]) -> Result<(), BoxError> {
        if bytes.len() as u64 > self.size {
            return Err("GPU buffer upload exceeds buffer size".into());
        }
        let allocation = self.allocation.as_mut().ok_or("unallocated buffer")?;
        allocation
            .mapped_slice_mut()
            .ok_or("buffer is not CPU-mapped")?[..bytes.len()]
            .copy_from_slice(bytes);
        if !allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
            && !bytes.is_empty()
        {
            // SAFETY: mapped allocation is atom-aligned and padded at allocation time;
            // the entire allocator block is mapped. Range stays within this allocation.
            unsafe {
                self.memory.ctx.device.flush_mapped_memory_ranges(&[
                    vk::MappedMemoryRange::default()
                        .memory(allocation.memory())
                        .offset(allocation.offset())
                        .size(aligned_size(bytes.len() as u64, self.memory.atom_size)),
                ])?;
            }
        }
        Ok(())
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: owners wait for upload/frame completion or defer destruction until
        // the last referring submission completes. Null handles are legal.
        unsafe {
            self.memory.ctx.device.destroy_buffer(self.handle, None);
        }
        if let Some(allocation) = self.allocation.take() {
            self.memory.free(allocation);
        }
    }
}

pub(crate) struct ImageResource {
    memory: Rc<Memory>,
    pub image: vk::Image,
    pub view: vk::ImageView,
    allocation: Option<Allocation>,
}
impl ImageResource {
    pub fn new(
        memory: Rc<Memory>,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Self, BoxError> {
        Self::with_usage(
            memory,
            extent,
            format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            depth_range(),
        )
    }
    pub fn shadow(memory: Rc<Memory>, extent: vk::Extent2D) -> Result<Self, BoxError> {
        Self::with_usage(
            memory,
            extent,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            depth_range(),
        )
    }
    pub fn hdr(memory: Rc<Memory>, extent: vk::Extent2D) -> Result<Self, BoxError> {
        Self::with_usage(
            memory,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            crate::swapchain::color_range(),
        )
    }
    pub fn sampled(
        memory: Rc<Memory>,
        extent: vk::Extent2D,
        levels: u32,
        format: vk::Format,
    ) -> Result<Self, BoxError> {
        Self::with_usage(
            memory,
            extent,
            format,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
            crate::swapchain::color_range().level_count(levels),
        )
    }
    pub(crate) fn with_usage(
        memory: Rc<Memory>,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        range: vk::ImageSubresourceRange,
    ) -> Result<Self, BoxError> {
        let mut owner = Self {
            memory,
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            allocation: None,
        };
        // SAFETY: positive extent and compatible format/usage supplied by the caller;
        // every successful handle is immediately owned for partial failure cleanup.
        unsafe {
            owner.image = owner.memory.ctx.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(range.level_count)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?;
            let requirements = owner
                .memory
                .ctx
                .device
                .get_image_memory_requirements(owner.image);
            owner.allocation = Some(owner.memory.allocator.borrow_mut().allocate(
                &AllocationCreateDesc {
                    name: "depth",
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::DedicatedImage(owner.image),
                },
            )?);
            owner
                .memory
                .track(owner.allocation.as_ref().expect("allocated image"));
            let allocation = owner.allocation.as_ref().expect("allocated depth");
            owner.memory.ctx.device.bind_image_memory(
                owner.image,
                allocation.memory(),
                allocation.offset(),
            )?;
            owner.view = owner.memory.ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(owner.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(range),
                None,
            )?;
        }
        Ok(owner)
    }
}
impl Drop for ImageResource {
    fn drop(&mut self) {
        // SAFETY: resource owner waits for all referring submissions before release.
        unsafe {
            self.memory.ctx.device.destroy_image_view(self.view, None);
            self.memory.ctx.device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            self.memory.free(allocation);
        }
    }
}
pub(crate) fn depth_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noncoherent_flush_size_is_atom_aligned() {
        assert_eq!(aligned_size(112, 256), 256);
        assert_eq!(aligned_size(256, 256), 256);
        assert_eq!(aligned_size(257, 256), 512);
    }
}

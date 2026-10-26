//! Vulkan-only bootstrap renderer. All FFI and GPU ownership stay in this crate.
mod context;
mod frame;
mod memory;
mod mesh;
mod pipeline;
mod scene_gpu;
mod swapchain;
mod texture;
use ash::vk;
use context::DeviceContext;
pub use context::{SurfaceTarget, ValidationStats};
use frame::Frame;
use govno_core::BoxError;
use govno_render::{FrameStatus, MeshData, MeshHandle, RenderExtent, RenderWorld};
use memory::Memory;
use mesh::{GpuMesh, RetirementQueue};
use scene_gpu::SceneGpu;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_RENDERER: AtomicU64 = AtomicU64::new(1);
use std::{marker::PhantomData, rc::Rc, sync::Arc};
use swapchain::Swapchain;

pub struct VulkanRenderer {
    ctx: Arc<DeviceContext>,
    memory: Rc<Memory>,
    scene_gpu: SceneGpu,
    meshes: HashMap<MeshHandle, GpuMesh>,
    retired: RetirementQueue<GpuMesh>,
    renderer_id: u64,
    next_mesh: u64,
    submitted: u64,
    completed: u64,
    frames: Vec<Frame>,
    swapchain: Option<Swapchain>,
    extent: RenderExtent,
    dirty: bool,
    frame_index: usize,
    failed: bool,
    // Queue/pool access remains on the thread owning the desktop event loop.
    _main_thread: PhantomData<Rc<()>>,
}
impl VulkanRenderer {
    pub fn new(
        window: Arc<dyn SurfaceTarget>,
        extent: RenderExtent,
        validation: bool,
        stats: Arc<ValidationStats>,
    ) -> Result<Self, BoxError> {
        let ctx = context::create_device(window, validation, stats)?;
        let memory = Memory::new(ctx.clone())?;
        let scene_gpu = SceneGpu::new(memory.clone(), 2)?;
        let mut renderer = Self {
            ctx,
            memory,
            scene_gpu,
            meshes: HashMap::new(),
            retired: RetirementQueue::default(),
            renderer_id: NEXT_RENDERER.fetch_add(1, Ordering::Relaxed),
            next_mesh: 0,
            submitted: 0,
            completed: 0,
            frames: Vec::new(),
            swapchain: None,
            extent,
            dirty: true,
            frame_index: 0,
            failed: false,
            _main_thread: PhantomData,
        };
        for _ in 0..2 {
            renderer.frames.push(Frame::new(renderer.ctx.clone())?);
        }
        renderer.recreate()?;
        Ok(renderer)
    }
    /// Initial API: one shared base-color image for all instances. Synchronous
    /// replacement is safe during rendering but deliberately waits for GPU users.
    pub fn set_base_color_texture(
        &mut self,
        data: &govno_render::TextureData,
    ) -> Result<(), BoxError> {
        data.validate()?;
        if self.failed {
            return Err("renderer has failed".into());
        }
        self.wait_idle()?;
        self.scene_gpu.set_texture(data)
    }
    pub fn upload_texture(
        &mut self,
        data: &govno_render::TextureData,
    ) -> Result<govno_render::TextureHandle, BoxError> {
        if self.failed {
            return Err("renderer has failed".into());
        }
        let handle = govno_render::TextureHandle::from_parts(self.renderer_id, self.next_mesh);
        self.next_mesh = self
            .next_mesh
            .checked_add(1)
            .ok_or("resource ID exhausted")?;
        self.scene_gpu.upload_texture(handle, data)?;
        Ok(handle)
    }
    pub fn remove_texture(&mut self, handle: govno_render::TextureHandle) -> Result<(), BoxError> {
        self.wait_idle()?;
        self.scene_gpu.remove_texture(handle)
    }
    pub fn upload_mesh(&mut self, data: &MeshData) -> Result<MeshHandle, BoxError> {
        if self.failed {
            return Err("renderer has failed".into());
        }
        let mesh = GpuMesh::upload(self.memory.clone(), data)?;
        let handle = MeshHandle::from_parts(self.renderer_id, self.next_mesh);
        self.next_mesh = self
            .next_mesh
            .checked_add(1)
            .ok_or("mesh identifier exhausted")?;
        self.meshes.insert(handle, mesh);
        Ok(handle)
    }
    pub fn remove_mesh(&mut self, handle: MeshHandle) -> Result<(), BoxError> {
        let mesh = self
            .meshes
            .remove(&handle)
            .ok_or("stale or foreign mesh handle")?;
        self.retired.retire(self.submitted, mesh);
        self.retired.collect(self.completed);
        Ok(())
    }
    pub fn resize(&mut self, extent: RenderExtent) {
        if self.extent == extent {
            return;
        }
        self.extent = extent;
        self.dirty = true;
    }

    fn recreate(&mut self) -> Result<(), BoxError> {
        if self.extent.is_zero() {
            return Ok(());
        }
        if let Some(chain) = &mut self.swapchain {
            chain.quiesce()?;
        }
        let old = self
            .swapchain
            .as_ref()
            .map_or(vk::SwapchainKHR::null(), |s| s.handle);
        let new = Swapchain::new(self.memory.clone(), self.extent, old, self.scene_gpu.layout)?;
        if let Some(new) = new {
            self.swapchain = Some(new);
            self.dirty = false;
        }
        Ok(())
    }
    /// An error poisons this renderer; continuing after a failed submission could
    /// otherwise wait forever on a reset fence. The caller must tear it down.
    pub fn draw(&mut self, world: &RenderWorld) -> Result<FrameStatus, BoxError> {
        if self.failed {
            return Err("renderer is stopped after an earlier failure".into());
        }
        if self.extent.is_zero() {
            return Ok(FrameStatus::Skipped);
        }
        world.validate(self.extent)?;
        for item in &world.instances {
            self.scene_gpu
                .descriptor(self.frame_index, item.material.base_color_texture)?;
        }
        if world
            .instances
            .iter()
            .any(|instance| !self.meshes.contains_key(&instance.mesh))
        {
            return Err("render world contains a stale or foreign mesh handle".into());
        }
        let result = self.draw_inner(world);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn draw_inner(&mut self, world: &RenderWorld) -> Result<FrameStatus, BoxError> {
        if self.extent.is_zero() {
            return Ok(FrameStatus::Skipped);
        }
        if self.dirty {
            self.recreate()?;
        }
        if self.dirty {
            return Ok(FrameStatus::Skipped);
        }
        let Some(chain) = &mut self.swapchain else {
            return Ok(FrameStatus::Skipped);
        };
        let frame = &mut self.frames[self.frame_index];
        let device = &self.ctx.device;
        // SAFETY: all queue operations are serialized by &mut self on the owning
        // thread. Frame fences protect command-pool/acquire semaphore reuse;
        // finished semaphores are indexed by acquired image, not frame index.
        unsafe {
            device.wait_for_fences(&[frame.fence], true, u64::MAX)?;
            self.completed = self.completed.max(frame.serial);
            self.retired.collect(self.completed);
            self.scene_gpu.update(
                self.frame_index,
                world,
                RenderExtent {
                    width: chain.extent.width,
                    height: chain.extent.height,
                },
                chain.srgb_attachment,
            )?;
            let (index, suboptimal) = match self.ctx.swapchain.acquire_next_image(
                chain.handle,
                1_000_000_000,
                frame.acquired,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.dirty = true;
                    return Ok(FrameStatus::Skipped);
                }
                Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                    return Ok(FrameStatus::Skipped);
                }
                Err(error) => return Err(error.into()),
            };
            let i = index as usize;
            self.dirty |= suboptimal;
            if self.ctx.present_fences {
                if chain.pending_present[i] {
                    device.wait_for_fences(&[chain.present_fences[i]], true, u64::MAX)?;
                    chain.pending_present[i] = false;
                }
                device.reset_fences(&[chain.present_fences[i]])?;
            }
            frame.record(
                chain,
                i,
                self.frame_index,
                &self.scene_gpu,
                world,
                &self.meshes,
            )?;
            // Reset only when work is ready to submit, never before acquisition.
            device.reset_fences(&[frame.fence])?;
            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.acquired)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let signal = [vk::SemaphoreSubmitInfo::default()
                .semaphore(chain.finished[i])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.command)];
            device.queue_submit2(
                self.ctx.queue,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait)
                    .command_buffer_infos(&commands)
                    .signal_semaphore_infos(&signal)],
                frame.fence,
            )?;
            self.submitted += 1;
            frame.serial = self.submitted;
            let waits = [chain.finished[i]];
            let chains = [chain.handle];
            let indices = [index];
            let fences = [if self.ctx.present_fences {
                chain.present_fences[i]
            } else {
                vk::Fence::null()
            }];
            let mut fence_info = vk::SwapchainPresentFenceInfoEXT::default().fences(&fences);
            let mut present = vk::PresentInfoKHR::default()
                .wait_semaphores(&waits)
                .swapchains(&chains)
                .image_indices(&indices);
            if self.ctx.present_fences {
                present = present.push_next(&mut fence_info);
            }
            let result = self.ctx.swapchain.queue_present(self.ctx.queue, &present);
            // Successful and WSI-rejected presents enqueue semaphore waits
            // and the maintenance1 fence signal operation.
            if self.ctx.present_fences && present_was_enqueued(result) {
                chain.pending_present[i] = true;
            }
            self.frame_index = (self.frame_index + 1) % self.frames.len();
            match result {
                Ok(suboptimal) => {
                    self.dirty |= suboptimal;
                    Ok(FrameStatus::Presented)
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.dirty = true;
                    Ok(FrameStatus::Skipped)
                }
                Err(error) => Err(error.into()),
            }
        }
    }
    pub fn wait_idle(&mut self) -> Result<(), BoxError> {
        if let Some(chain) = &mut self.swapchain {
            chain.quiesce()?;
        }
        self.ctx.wait_idle()?;
        self.completed = self.submitted;
        self.retired.collect(self.completed);
        Ok(())
    }
}
impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        if let Err(error) = self.wait_idle() {
            tracing::error!(%error, "renderer cleanup wait failed");
        }
        self.swapchain.take();
        self.frames.clear();
        self.meshes.clear();
        self.retired.collect(u64::MAX);
    }
}

// WSI rejection still enqueues queue operations. Allocation errors do not.
fn present_was_enqueued(result: Result<bool, vk::Result>) -> bool {
    matches!(
        result,
        Ok(_)
            | Err(vk::Result::ERROR_OUT_OF_DATE_KHR
                | vk::Result::ERROR_SURFACE_LOST_KHR
                | vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wsi_rejection_still_requires_presentation_fence_wait() {
        for result in [
            Ok(false),
            Ok(true),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR),
            Err(vk::Result::ERROR_SURFACE_LOST_KHR),
            Err(vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT),
        ] {
            assert!(present_was_enqueued(result));
        }
    }
    #[test]
    fn failed_enqueue_must_not_wait_on_unsignaled_present_fence() {
        for error in [
            vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            vk::Result::ERROR_DEVICE_LOST,
        ] {
            assert!(!present_was_enqueued(Err(error)));
        }
    }
}

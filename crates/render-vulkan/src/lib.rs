//! Vulkan-only bootstrap renderer. All FFI and GPU ownership stay in this crate.
mod capabilities;
mod compute;
mod timeline;
pub use capabilities::{DeviceCapabilities, DeviceProfile};
mod context;
mod effects;
mod frame;
mod memory;
mod mesh;
mod metrics;
mod pipeline;
mod post;
mod reflection;
mod scene_gpu;
mod shadow;
mod specular;
mod swapchain;
mod sync;
mod texture;
use ash::vk;
use context::DeviceContext;
pub use context::{SurfaceTarget, ValidationStats};
pub use frame::DrawCounts;
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
    timeline: timeline::Timeline,
    measurements: metrics::Measurements,
    completed: u64,
    frames: Vec<Frame>,
    swapchain: Option<Swapchain>,
    extent: RenderExtent,
    dirty: bool,
    frame_index: usize,
    failed: bool,
    pub last_draw_counts: DrawCounts,
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
        let timeline = timeline::Timeline::new(ctx.clone())?;
        let mut renderer = Self {
            ctx,
            memory,
            scene_gpu,
            meshes: HashMap::new(),
            retired: RetirementQueue::default(),
            renderer_id: NEXT_RENDERER.fetch_add(1, Ordering::Relaxed),
            next_mesh: 0,
            submitted: 0,
            timeline,
            measurements: metrics::Measurements::new(),
            completed: 0,
            frames: Vec::new(),
            swapchain: None,
            extent,
            dirty: true,
            frame_index: 0,
            failed: false,
            last_draw_counts: DrawCounts::default(),
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
        if data.color_space != govno_render::TextureColorSpace::Srgb {
            return Err("shared base-color texture must be sRGB".into());
        }
        if self.failed {
            return Err("renderer has failed".into());
        }
        self.wait_idle()?;
        self.scene_gpu.set_texture(data)
    }
    pub fn start_benchmark(&mut self) {
        self.measurements.enable();
        for frame in &mut self.frames {
            frame.enable_timing();
        }
    }
    pub fn benchmark_report(&mut self, warmup: usize) -> Result<String, BoxError> {
        if !self.measurements.enabled {
            return Err("benchmarking was not enabled".into());
        }
        self.wait_idle()?;
        for frame in &mut self.frames {
            if let Some(timing) = frame.timing()? {
                self.measurements.gpu_sample(timing);
            }
        }
        let cpu = &self.measurements.cpu[warmup.min(self.measurements.cpu.len())..];
        let gpu = &self.measurements.gpu[warmup.min(self.measurements.gpu.len())..];
        let profile = &self.ctx.capabilities;
        let report = serde_json::json!({"schema":1,"gpu":profile.name,"vendor_id":profile.vendor_id,"device_id":profile.device_id,"api_version":profile.api_version,"driver_version":profile.driver_version,"profiles":{"modern":profile.modern,"advanced":profile.advanced,"ray_tracing":profile.ray_tracing},"warmup_frames":warmup,"presented_samples":self.measurements.cpu.len(),"gpu_samples":self.measurements.gpu.len(),"cpu_draw_api":metrics::distribution(cpu),"cpu_command_record":metrics::distribution(&self.measurements.record[warmup.min(self.measurements.record.len())..]),"gpu_frame":metrics::distribution(&gpu.iter().map(|v|v[0]).collect::<Vec<_>>()),"gpu_shadow":metrics::distribution(&gpu.iter().map(|v|v[1]).collect::<Vec<_>>()),"gpu_forward":metrics::distribution(&gpu.iter().map(|v|v[2]).collect::<Vec<_>>()),"gpu_post":metrics::distribution(&gpu.iter().map(|v|v[3]).collect::<Vec<_>>()),"draws":{"main":self.measurements.submitted,"culled":self.measurements.culled,"shadow":self.measurements.shadow},"memory":self.memory.report(),"pipeline_creation":{"count":self.ctx.pipeline_builds.load(Ordering::Relaxed),"total_ms":self.ctx.pipeline_build_ns.load(Ordering::Relaxed)as f64*1e-6,"cache_loaded_bytes":self.ctx.pipeline_cache_loaded_bytes},"timeline":{"submitted":self.submitted,"completed":self.completed},"notes":"CPU = draw API wall time incl acquire/waits/submit/present, includes validation/material preparation; excludes app/event-loop work. GPU = timestamped command buffer, excludes presentation. Memory allocator counters exclude driver/WSI; optional EXT_memory_budget provides driver heap estimates. CPU command-record excludes waits and app validation. Profiles report support; optional features are not enabled."});
        Ok(serde_json::to_string_pretty(&report)?)
    }
    pub fn device_capabilities(&self) -> &DeviceCapabilities {
        &self.ctx.capabilities
    }
    pub fn verify_environment(
        &mut self,
        data: &govno_render::EnvironmentRadiance,
    ) -> Result<[f32; 3], BoxError> {
        self.wait_idle()?;
        self.scene_gpu.verify_environment(data)
    }
    pub fn post_diagnostics(&mut self) -> Result<Option<[f32; 4]>, BoxError> {
        self.wait_idle()?;
        let ctx = self.ctx.clone();
        crate::mesh::Upload::run(ctx.clone(), |cmd| {
            crate::sync::memory(
                &ctx,
                cmd,
                (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_WRITE,
                ),
                (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
            )
        })?;
        if let Some(effects) = self
            .swapchain
            .as_mut()
            .and_then(|chain| chain.post.effects.as_mut())
        {
            let bytes = effects.exposure.read()?;
            let mut values = [0.0; 4];
            for (i, b) in bytes.as_chunks::<4>().0.iter().enumerate() {
                values[i] = f32::from_ne_bytes(*b);
            }
            return Ok(Some(values));
        }
        Ok(None)
    }
    pub fn set_environment_gpu(
        &mut self,
        data: &govno_render::EnvironmentRadiance,
        probe: u32,
    ) -> Result<(), BoxError> {
        if self.failed {
            return Err("renderer has failed".into());
        }
        self.wait_idle()?;
        self.scene_gpu.set_environment_gpu(data, probe)
    }
    /// Replace the one global prefiltered environment; synchronous load-time API.
    pub fn set_specular_environment(
        &mut self,
        data: &govno_render::SpecularEnvironment,
    ) -> Result<(), BoxError> {
        if self.failed {
            return Err("renderer has failed".into());
        }
        self.wait_idle()?;
        self.scene_gpu.set_specular_environment(data)
    }
    pub fn upload_texture(
        &mut self,
        data: &govno_render::TextureData,
    ) -> Result<govno_render::TextureHandle, BoxError> {
        self.upload_texture_with_sampling(data, Default::default())
    }
    pub fn upload_texture_with_sampling(
        &mut self,
        data: &govno_render::TextureData,
        sampling: govno_render::TextureSampling,
    ) -> Result<govno_render::TextureHandle, BoxError> {
        if self.failed {
            return Err("renderer has failed".into());
        }
        let handle = govno_render::TextureHandle::from_parts(self.renderer_id, self.next_mesh);
        self.next_mesh = self
            .next_mesh
            .checked_add(1)
            .ok_or("resource ID exhausted")?;
        self.scene_gpu.upload_texture(handle, data, sampling)?;
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
        let started = std::time::Instant::now();
        if self.failed {
            return Err("renderer is stopped after an earlier failure".into());
        }
        if self.extent.is_zero() {
            return Ok(FrameStatus::Skipped);
        }
        world.validate(self.extent)?;
        self.scene_gpu.collect_descriptors(self.completed);
        self.scene_gpu.prepare(world, self.submitted)?;
        for item in &world.instances {
            self.scene_gpu
                .descriptors(self.frame_index, &item.material)?;
        }
        if world
            .instances
            .iter()
            .any(|instance| !self.meshes.contains_key(&instance.mesh))
        {
            return Err("render world contains a stale or foreign mesh handle".into());
        }
        let result = self.draw_inner(world);
        if self.measurements.enabled
            && matches!(result, Ok(FrameStatus::Presented))
            && self.measurements.cpu.len() < 100000
        {
            self.measurements
                .cpu
                .push(started.elapsed().as_secs_f64() * 1000.0);
            self.measurements.submitted += u64::from(self.last_draw_counts.submitted);
            self.measurements.culled += u64::from(self.last_draw_counts.culled);
            self.measurements.shadow += u64::from(self.last_draw_counts.shadow);
        }
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
        // thread. Frame timeline completion protects command-pool/acquire semaphore reuse;
        // finished semaphores are indexed by acquired image, not frame index.
        unsafe {
            self.timeline.wait(frame.serial)?;
            if let Some(timing) = frame.timing()? {
                self.measurements.gpu_sample(timing);
            }
            self.completed = self.timeline.completed()?;
            self.retired.collect(self.completed);
            self.scene_gpu.collect_descriptors(self.completed);
            self.scene_gpu.update(
                self.frame_index,
                world,
                RenderExtent {
                    width: chain.extent.width,
                    height: chain.extent.height,
                },
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
            let record_started = self.measurements.enabled.then(std::time::Instant::now);
            self.last_draw_counts = frame.record(
                chain,
                i,
                self.frame_index,
                &self.scene_gpu,
                world,
                &self.meshes,
            )?;
            if let Some(start) = record_started
                && self.measurements.record.len() < 100000
            {
                self.measurements
                    .record
                    .push(start.elapsed().as_secs_f64() * 1000.0);
            }
            // Signal a fresh completion value only when recorded work is ready.
            let next_serial = self
                .submitted
                .checked_add(1)
                .ok_or("submission serial exhausted")?;
            let wait = [vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.acquired)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            let signal = [
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(chain.finished[i])
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.timeline.semaphore)
                    .value(next_serial)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            ];
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.command)];
            device.queue_submit2(
                self.ctx.queue,
                &[vk::SubmitInfo2::default()
                    .wait_semaphore_infos(&wait)
                    .command_buffer_infos(&commands)
                    .signal_semaphore_infos(&signal)],
                vk::Fence::null(),
            )?;
            self.submitted = next_serial;
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
        self.completed = self.timeline.completed()?;
        self.retired.collect(self.completed);
        self.scene_gpu.collect_descriptors(self.completed);
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

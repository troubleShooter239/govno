use crate::context::DeviceContext;
use ash::vk;
use govno_core::BoxError;
use std::sync::Arc;
pub(crate) struct Timeline {
    ctx: Arc<DeviceContext>,
    pub semaphore: vk::Semaphore,
}
impl Timeline {
    pub fn new(ctx: Arc<DeviceContext>) -> Result<Self, BoxError> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        // SAFETY: timeline feature enabled; returned handle receives immediate ownership.
        let semaphore = unsafe {
            ctx.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
                None,
            )?
        };
        Ok(Self { ctx, semaphore })
    }
    pub fn completed(&self) -> Result<u64, BoxError> {
        // SAFETY: live semaphore; all host use is serialized on renderer thread.
        Ok(unsafe {
            self.ctx
                .device
                .get_semaphore_counter_value(self.semaphore)?
        })
    }
    pub fn wait(&self, value: u64) -> Result<(), BoxError> {
        if value == 0 {
            return Ok(());
        }
        // SAFETY: values are signaled by the renderer's monotonically ordered submissions.
        unsafe {
            self.ctx.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&[self.semaphore])
                    .values(&[value]),
                u64::MAX,
            )?;
        }
        Ok(())
    }
}
impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: renderer quiesces the queue before dropping this owner.
        unsafe {
            self.ctx.device.destroy_semaphore(self.semaphore, None);
        }
    }
}

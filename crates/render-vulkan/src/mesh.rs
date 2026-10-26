use crate::{
    context::DeviceContext,
    memory::{Buffer, Memory},
};
use ash::vk;
use govno_core::BoxError;
use govno_render::MeshData;
use gpu_allocator::MemoryLocation;
use std::{rc::Rc, sync::Arc};

pub(crate) struct GpuMesh {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub index_count: u32,
}
pub(crate) struct Upload {
    ctx: Arc<DeviceContext>,
    pool: vk::CommandPool,
    fence: vk::Fence,
    submitted: bool,
}
impl Drop for Upload {
    fn drop(&mut self) {
        // SAFETY: synchronous upload owns all pending commands. On failure after
        // submission, wait before destroying the pool/staging buffers; DEVICE_LOST
        // still permits resource destruction.
        unsafe {
            if self.submitted {
                let _ = self
                    .ctx
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX);
            }
            self.ctx.device.destroy_fence(self.fence, None);
            self.ctx.device.destroy_command_pool(self.pool, None);
        }
    }
}
impl GpuMesh {
    pub fn upload(memory: Rc<Memory>, mesh: &MeshData) -> Result<Self, BoxError> {
        mesh.validate()?;
        let vertices: Vec<u8> = mesh
            .vertices
            .iter()
            .flat_map(|v| {
                v.position
                    .into_iter()
                    .chain(v.normal)
                    .chain(v.uv)
                    .chain(v.tangent)
            })
            .flat_map(f32::to_ne_bytes)
            .collect();
        let indices: Vec<u8> = mesh.indices.iter().flat_map(|i| i.to_ne_bytes()).collect();
        let mut staging = Buffer::new(
            memory.clone(),
            (vertices.len() + indices.len()) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        let mut bytes = vertices.clone();
        bytes.extend_from_slice(&indices);
        staging.write(&bytes)?;
        let gpu = Self {
            vertices: Buffer::new(
                memory.clone(),
                vertices.len() as u64,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::GpuOnly,
            )?,
            indices: Buffer::new(
                memory.clone(),
                indices.len() as u64,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::GpuOnly,
            )?,
            index_count: mesh.indices.len() as u32,
        };
        let ctx = memory.ctx.clone();
        Upload::run(ctx.clone(), |command| {
            // SAFETY: staging and destination buffers remain live through upload
            // completion; validated ranges are aligned and within each buffer.
            unsafe {
                ctx.device.cmd_copy_buffer(
                    command,
                    staging.handle,
                    gpu.vertices.handle,
                    &[vk::BufferCopy::default().size(vertices.len() as u64)],
                );
                ctx.device.cmd_copy_buffer(
                    command,
                    staging.handle,
                    gpu.indices.handle,
                    &[vk::BufferCopy::default()
                        .src_offset(vertices.len() as u64)
                        .size(indices.len() as u64)],
                );
                let barriers = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                    .dst_access_mask(
                        vk::AccessFlags2::VERTEX_ATTRIBUTE_READ | vk::AccessFlags2::INDEX_READ,
                    )];
                ctx.device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().memory_barriers(&barriers),
                );
            }
        })?;
        tracing::info!(
            vertices = mesh.vertices.len(),
            indices = mesh.indices.len(),
            "mesh uploaded"
        );
        Ok(gpu)
    }
}

/// Resources are kept alive until the last referring single-queue submission
/// completes. This queue contains owned objects, not callbacks with hidden captures.
pub(crate) struct RetirementQueue<T>(Vec<(u64, T)>);
impl<T> Default for RetirementQueue<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}
impl<T> RetirementQueue<T> {
    pub fn retire(&mut self, serial: u64, resource: T) {
        self.0.push((serial, resource));
    }
    pub fn collect(&mut self, completed: u64) {
        self.0.retain(|(serial, _)| *serial > completed);
    }
}
impl Upload {
    pub fn run(
        ctx: Arc<DeviceContext>,
        record: impl FnOnce(vk::CommandBuffer),
    ) -> Result<(), BoxError> {
        let mut owner = Self {
            ctx,
            pool: vk::CommandPool::null(),
            fence: vk::Fence::null(),
            submitted: false,
        };
        // SAFETY: command pool is owned until fence completion; caller keeps all
        // resources captured by record alive until this synchronous method returns.
        unsafe {
            owner.pool = owner.ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(owner.ctx.family),
                None,
            )?;
            owner.fence = owner
                .ctx
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmd = owner.ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(owner.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            owner.ctx.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(cmd);
            owner.ctx.device.end_command_buffer(cmd)?;
            let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            owner.ctx.device.queue_submit2(
                owner.ctx.queue,
                &[vk::SubmitInfo2::default().command_buffer_infos(&commands)],
                owner.fence,
            )?;
            owner.submitted = true;
            owner
                .ctx
                .device
                .wait_for_fences(&[owner.fence], true, u64::MAX)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    struct Probe(Rc<Cell<usize>>);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    #[test]
    fn retirement_waits_for_last_use_and_drops_once() {
        let count = Rc::new(Cell::new(0));
        let mut queue = RetirementQueue::default();
        queue.retire(4, Probe(count.clone()));
        queue.retire(7, Probe(count.clone()));
        queue.collect(3);
        assert_eq!(count.get(), 0);
        queue.collect(4);
        assert_eq!(count.get(), 1);
        queue.collect(6);
        assert_eq!(count.get(), 1);
        queue.collect(7);
        assert_eq!(count.get(), 2);
        queue.collect(100);
        assert_eq!(count.get(), 2);
    }
}

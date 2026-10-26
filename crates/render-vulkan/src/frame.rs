use crate::{
    context::DeviceContext,
    memory::depth_range,
    mesh::GpuMesh,
    scene_gpu::draw_constants,
    swapchain::{Swapchain, color_range},
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{MeshHandle, RenderWorld};
use std::{collections::HashMap, sync::Arc};

pub(crate) struct Frame {
    pub ctx: Arc<DeviceContext>,
    pub pool: vk::CommandPool,
    pub command: vk::CommandBuffer,
    pub acquired: vk::Semaphore,
    pub fence: vk::Fence,
    pub serial: u64,
}
impl Frame {
    pub fn new(ctx: Arc<DeviceContext>) -> Result<Self, BoxError> {
        let mut frame = Self {
            ctx,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            acquired: vk::Semaphore::null(),
            fence: vk::Fence::null(),
            serial: 0,
        };
        // SAFETY: every created resource is recorded before further fallible calls.
        unsafe {
            frame.pool = frame.ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(frame.ctx.family),
                None,
            )?;
            frame.command = frame.ctx.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(frame.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            frame.acquired = frame
                .ctx
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            frame.fence = frame.ctx.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?;
        }
        Ok(frame)
    }
    pub fn record(
        &self,
        chain: &Swapchain,
        image_index: usize,
        frame_index: usize,
        scene: &crate::scene_gpu::SceneGpu,
        world: &RenderWorld,
        meshes: &HashMap<MeshHandle, GpuMesh>,
    ) -> Result<(), BoxError> {
        let device = &self.ctx.device;
        let image = chain.images[image_index];
        let depth = &chain.depth[frame_index];
        let extent = chain.extent;
        // SAFETY: frame fence has completed, so its command pool, uniform and depth
        // target are no longer in flight. Meshes are validated and retained through
        // the submission serial. Swapchain image has been acquired.
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color_range()),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(depth.image)
                    .subresource_range(depth_range()),
            ];
            device.cmd_pipeline_barrier2(
                self.command,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            let colors = [vk::RenderingAttachmentInfo::default()
                .image_view(chain.views[image_index])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.018, 0.025, 0.045, 1.0],
                    },
                })];
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            device.cmd_begin_rendering(
                self.command,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent,
                    })
                    .layer_count(1)
                    .color_attachments(&colors)
                    .depth_attachment(&depth_attachment),
            );
            device.cmd_bind_pipeline(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                chain.pipeline.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::GRAPHICS,
                chain.pipeline.layout,
                0,
                &[scene.descriptor(frame_index, None)?],
                &[],
            );
            device.cmd_set_viewport(
                self.command,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                self.command,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                }],
            );
            for instance in &world.instances {
                device.cmd_bind_descriptor_sets(
                    self.command,
                    vk::PipelineBindPoint::GRAPHICS,
                    chain.pipeline.layout,
                    0,
                    &[scene.descriptor(frame_index, instance.material.base_color_texture)?],
                    &[],
                );
                let mesh = meshes
                    .get(&instance.mesh)
                    .ok_or("stale or foreign mesh handle")?;
                device.cmd_bind_vertex_buffers(self.command, 0, &[mesh.vertices.handle], &[0]);
                device.cmd_bind_index_buffer(
                    self.command,
                    mesh.indices.handle,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_push_constants(
                    self.command,
                    chain.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &draw_constants(instance),
                );
                device.cmd_draw_indexed(self.command, mesh.index_count, 1, 0, 0, 0);
            }
            device.cmd_end_rendering(self.command);
            let to_present = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())];
            device.cmd_pipeline_barrier2(
                self.command,
                &vk::DependencyInfo::default().image_memory_barriers(&to_present),
            );
            device.end_command_buffer(self.command)?;
        }
        Ok(())
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        // SAFETY: renderer waits idle before dropping frames; construction failures
        // have no submitted work. Null handles are permitted for destruction.
        unsafe {
            self.ctx.device.destroy_fence(self.fence, None);
            self.ctx.device.destroy_semaphore(self.acquired, None);
            self.ctx.device.destroy_command_pool(self.pool, None);
        }
    }
}

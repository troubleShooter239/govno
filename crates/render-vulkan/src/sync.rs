use crate::context::DeviceContext;
use ash::vk;
pub(crate) fn transition(
    ctx: &DeviceContext,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2, vk::ImageLayout),
) {
    let barriers = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .old_layout(src.2)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)
        .new_layout(dst.2)
        .image(image)
        .subresource_range(range)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
    // SAFETY: caller records on the owning queue; image/range/layouts describe live resources.
    unsafe {
        ctx.device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barriers),
        );
    }
}
pub(crate) fn memory(
    ctx: &DeviceContext,
    cmd: vk::CommandBuffer,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let barriers = [vk::MemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)];
    // SAFETY: caller records this dependency in a live command buffer on the owning queue.
    unsafe {
        ctx.device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&barriers),
        );
    }
}

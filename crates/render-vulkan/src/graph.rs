use crate::{context::DeviceContext, memory::ImageResource};
use ash::vk;
use govno_core::BoxError;
use govno_render_graph::{Access, CompiledGraph, CompiledPass, Graph, Lifetime, Use};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResourceState {
    stages: vk::PipelineStageFlags2,
    access: vk::AccessFlags2,
    // Ignored for buffer bindings.
    layout: vk::ImageLayout,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Pass {
    ResetCull,
    Cull,
    ReadbackCull,
    HostReadback,
    Shadow,
    Forward,
    Effects(crate::effects::EffectPass),
    Composite,
    Present,
}
#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Resource {
    Hdr,
    Depth,
    Output,
    Shadow,
    Bloom,
    Scratch,
    Exposure,
    Objects,
    CullInput,
    Commands,
    Counts,
    Readback,
}
pub(crate) const RESOURCE_COUNT: usize = 12;
pub(crate) struct FrameGraph {
    pub plan: CompiledGraph<ResourceState, Pass>,
    pub lifetimes: [Option<Lifetime>; RESOURCE_COUNT],
}
impl FrameGraph {
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"passes":self.plan.passes.iter().map(|p|p.name.as_str()).collect::<Vec<_>>(),
            "resources":(["hdr","depth","swapchain","shadow","bloom","scratch","exposure","objects","cull_input","indirect_commands","draw_counts","draw_readback"]
                .into_iter().zip(&self.lifetimes).map(|(name,lifetime)|
                serde_json::json!({"name":name,"first":lifetime.map(|l|l.first),"last":lifetime.map(|l|l.last)})).collect::<Vec<_>>())})
    }
}
pub(crate) fn compile(gpu_culling: bool) -> Result<FrameGraph, BoxError> {
    use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
    let state = |stages, access, layout| ResourceState {
        stages,
        access,
        layout,
    };
    let undefined = state(S::NONE, A::NONE, L::UNDEFINED);
    let color = state(
        S::COLOR_ATTACHMENT_OUTPUT,
        A::COLOR_ATTACHMENT_WRITE,
        L::COLOR_ATTACHMENT_OPTIMAL,
    );
    let depth = state(
        S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
        A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
        L::DEPTH_ATTACHMENT_OPTIMAL,
    );
    let sampled = state(
        S::COMPUTE_SHADER | S::FRAGMENT_SHADER,
        A::SHADER_SAMPLED_READ,
        L::SHADER_READ_ONLY_OPTIMAL,
    );
    let shadow_read = state(
        S::FRAGMENT_SHADER,
        A::SHADER_SAMPLED_READ,
        L::DEPTH_READ_ONLY_OPTIMAL,
    );
    let storage_write = state(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, L::GENERAL);
    let storage_sampled = state(S::COMPUTE_SHADER, A::SHADER_SAMPLED_READ, L::GENERAL);
    let exposure_write = state(
        S::COMPUTE_SHADER,
        A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
        L::UNDEFINED,
    );
    let exposure_read = state(
        S::COMPUTE_SHADER | S::FRAGMENT_SHADER,
        A::SHADER_STORAGE_READ,
        L::UNDEFINED,
    );
    let present = state(S::NONE, A::NONE, L::PRESENT_SRC_KHR);
    let mut graph = Graph::default();
    // This order is the Resource enum and the frame's binding table.
    let hdr = graph.resource("HDR", undefined, false);
    let depth_image = graph.resource("Depth", undefined, false);
    // Layout transition must chain to the WSI acquire wait at COLOR_ATTACHMENT_OUTPUT.
    let output = graph.resource(
        "Acquired swapchain",
        state(S::COLOR_ATTACHMENT_OUTPUT, A::NONE, L::UNDEFINED),
        false,
    );
    let shadow = graph.resource("Cascaded shadow atlas", undefined, false);
    let bloom = graph.resource("Bloom", undefined, false);
    let scratch = graph.resource("Bloom scratch", undefined, false);
    // Persistent exposure is shared across frames: include prior compute writes and
    // fragment reads, plus the initial host upload. Queue order alone is insufficient.
    let exposure = graph.resource(
        "Persistent exposure",
        state(
            S::HOST | S::COMPUTE_SHADER | S::FRAGMENT_SHADER,
            A::HOST_WRITE | A::SHADER_STORAGE_READ | A::SHADER_STORAGE_WRITE,
            L::UNDEFINED,
        ),
        true,
    );
    let objects = graph.resource(
        "GPU Scene objects",
        state(S::HOST, A::HOST_WRITE, L::UNDEFINED),
        true,
    );
    let input = graph.resource(
        "Culling input",
        state(S::HOST, A::HOST_WRITE, L::UNDEFINED),
        true,
    );
    let commands = graph.resource("Indirect commands", undefined, false);
    let counts = graph.resource("Batch counts", undefined, false);
    let readback = graph.resource(
        "Culling readback",
        state(S::HOST, A::HOST_READ, L::UNDEFINED),
        true,
    );
    let compute_read = state(S::COMPUTE_SHADER, A::SHADER_STORAGE_READ, L::UNDEFINED);
    let indirect_read = state(S::DRAW_INDIRECT, A::INDIRECT_COMMAND_READ, L::UNDEFINED);
    let usage = |resource, state, access| Use {
        resource,
        state,
        access,
    };
    if gpu_culling {
        graph.pass(
            "Reset culling counts",
            Pass::ResetCull,
            vec![usage(
                counts,
                state(S::TRANSFER, A::TRANSFER_WRITE, L::UNDEFINED),
                Access::Write,
            )],
        );
        graph.pass(
            "Frustum culling and command compaction",
            Pass::Cull,
            vec![
                usage(input, compute_read, Access::Read),
                usage(objects, compute_read, Access::Read),
                usage(
                    commands,
                    state(S::COMPUTE_SHADER, A::SHADER_STORAGE_WRITE, L::UNDEFINED),
                    Access::Write,
                ),
                usage(counts, exposure_write, Access::ReadWrite),
            ],
        );
    }
    graph.pass(
        "Cascaded shadow atlas",
        Pass::Shadow,
        vec![
            usage(shadow, depth, Access::Write),
            usage(
                objects,
                state(
                    S::VERTEX_SHADER | S::FRAGMENT_SHADER,
                    A::SHADER_STORAGE_READ,
                    L::UNDEFINED,
                ),
                Access::Read,
            ),
        ],
    );
    let mut forward_uses = vec![
        usage(hdr, color, Access::Write),
        usage(depth_image, depth, Access::Write),
        usage(shadow, shadow_read, Access::Read),
        usage(
            objects,
            state(
                S::VERTEX_SHADER | S::FRAGMENT_SHADER,
                A::SHADER_STORAGE_READ,
                L::UNDEFINED,
            ),
            Access::Read,
        ),
    ];
    if gpu_culling {
        forward_uses.extend([
            usage(commands, indirect_read, Access::Read),
            usage(counts, indirect_read, Access::Read),
        ]);
    }
    graph.pass("Forward PBR", Pass::Forward, forward_uses);
    graph.pass(
        "Exposure",
        Pass::Effects(crate::effects::EffectPass::Exposure),
        vec![
            usage(hdr, sampled, Access::Read),
            usage(exposure, exposure_write, Access::ReadWrite),
        ],
    );
    graph.pass(
        "Bloom extraction",
        Pass::Effects(crate::effects::EffectPass::Extract),
        vec![
            usage(hdr, sampled, Access::Read),
            usage(exposure, exposure_read, Access::Read),
            usage(bloom, storage_write, Access::Write),
        ],
    );
    graph.pass(
        "Bloom horizontal",
        Pass::Effects(crate::effects::EffectPass::Horizontal),
        vec![
            usage(bloom, storage_sampled, Access::Read),
            usage(scratch, storage_write, Access::Write),
        ],
    );
    graph.pass(
        "Bloom vertical",
        Pass::Effects(crate::effects::EffectPass::Vertical),
        vec![
            usage(scratch, storage_sampled, Access::Read),
            usage(bloom, storage_write, Access::Write),
        ],
    );
    graph.pass(
        "Tone mapping and UI",
        Pass::Composite,
        vec![
            usage(hdr, sampled, Access::Read),
            usage(bloom, sampled, Access::Read),
            usage(exposure, exposure_read, Access::Read),
            usage(output, color, Access::Write),
        ],
    );
    if gpu_culling {
        graph.pass(
            "Copy culling counters",
            Pass::ReadbackCull,
            vec![
                usage(
                    counts,
                    state(S::TRANSFER, A::TRANSFER_READ, L::UNDEFINED),
                    Access::Read,
                ),
                usage(
                    readback,
                    state(S::TRANSFER, A::TRANSFER_WRITE, L::UNDEFINED),
                    Access::Write,
                ),
            ],
        );
        graph.pass(
            "Culling host readback",
            Pass::HostReadback,
            vec![usage(
                readback,
                state(S::HOST, A::HOST_READ, L::UNDEFINED),
                Access::Read,
            )],
        );
    }
    graph.pass(
        "Present",
        Pass::Present,
        vec![usage(output, present, Access::Read)],
    );
    let resources = [
        hdr,
        depth_image,
        output,
        shadow,
        bloom,
        scratch,
        exposure,
        objects,
        input,
        commands,
        counts,
        readback,
    ];
    let plan = graph.compile()?;
    let lifetimes = resources.map(|r| plan.lifetime(r));
    tracing::info!(
        passes = plan.passes.len(),
        ?lifetimes,
        "frame render graph compiled"
    );
    Ok(FrameGraph { plan, lifetimes })
}
#[derive(Clone, Copy)]
pub(crate) enum Binding {
    Image {
        handle: vk::Image,
        range: vk::ImageSubresourceRange,
    },
    Buffer {
        handle: vk::Buffer,
        size: u64,
    },
}
impl Binding {
    pub fn image(image: &ImageResource, range: vk::ImageSubresourceRange) -> Self {
        Self::Image {
            handle: image.image,
            range,
        }
    }
}
pub(crate) fn barriers(
    ctx: &DeviceContext,
    command: vk::CommandBuffer,
    pass: &CompiledPass<ResourceState, Pass>,
    bindings: &[Binding; RESOURCE_COUNT],
) {
    let mut images = [vk::ImageMemoryBarrier2::default(); RESOURCE_COUNT];
    let mut buffers = [vk::BufferMemoryBarrier2::default(); RESOURCE_COUNT];
    let (mut image_count, mut buffer_count) = (0, 0);
    for t in &pass.transitions {
        let stages = t
            .sources
            .iter()
            .fold(vk::PipelineStageFlags2::NONE, |all, s| all | s.stages);
        let access = t
            .sources
            .iter()
            .fold(vk::AccessFlags2::NONE, |all, s| all | s.access);
        match bindings[t.resource.index()] {
            Binding::Image { handle, range } => {
                images[image_count] = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(stages)
                    .src_access_mask(access)
                    .old_layout(t.before.layout)
                    .dst_stage_mask(t.after.stages)
                    .dst_access_mask(t.after.access)
                    .new_layout(t.after.layout)
                    .image(handle)
                    .subresource_range(range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
                image_count += 1;
            }
            Binding::Buffer { handle, size } => {
                buffers[buffer_count] = vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(stages)
                    .src_access_mask(access)
                    .dst_stage_mask(t.after.stages)
                    .dst_access_mask(t.after.access)
                    .buffer(handle)
                    .size(size)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
                buffer_count += 1;
            }
        }
    }
    if image_count + buffer_count > 0 {
        // SAFETY: live bindings belong to this frame/queue; graph scopes include all
        // writer/reader hazards, and the frame slot completed before recording.
        unsafe {
            ctx.device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(&images[..image_count])
                    .buffer_memory_barriers(&buffers[..buffer_count]),
            );
        }
    }
}

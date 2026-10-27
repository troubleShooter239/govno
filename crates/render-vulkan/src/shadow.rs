use crate::{
    context::DeviceContext,
    memory::{ImageResource, Memory, depth_range},
    mesh::GpuMesh,
    pipeline::ForwardPipeline,
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{MeshHandle, RenderWorld};
use std::{collections::HashMap, rc::Rc, sync::Arc};
pub(crate) const SHADOW_SIZE: u32 = 2048;
pub(crate) struct ShadowMap {
    pub images: Vec<ImageResource>,
    pub pipeline: ForwardPipeline,
    ctx: Arc<DeviceContext>,
}
impl ShadowMap {
    pub fn new(memory: Rc<Memory>) -> Result<Self, BoxError> {
        // SAFETY: physical device belongs to the retained live instance.
        let features = unsafe {
            memory
                .ctx
                .instance()
                .get_physical_device_format_properties(memory.ctx.physical, vk::Format::D32_SFLOAT)
        }
        .optimal_tiling_features;
        if !features.contains(
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::FormatFeatureFlags::SAMPLED_IMAGE,
        ) {
            return Err(
                "D32 depth attachment and sampled-image support required for shadows".into(),
            );
        }
        let mut owner = Self {
            images: vec![],
            pipeline: ForwardPipeline::shadow(memory.ctx.clone())?,
            ctx: memory.ctx.clone(),
        };
        for _ in 0..2 {
            owner.images.push(ImageResource::shadow(
                memory.clone(),
                vk::Extent2D {
                    width: SHADOW_SIZE * 2,
                    height: SHADOW_SIZE * 2,
                },
            )?);
        }
        tracing::info!(
            resolution = SHADOW_SIZE,
            "per-frame directional shadow maps created"
        );
        Ok(owner)
    }
    pub fn bind_images(&self, sets: &[vk::DescriptorSet]) {
        for (set, image) in sets.iter().zip(&self.images) {
            let info = [vk::DescriptorImageInfo::default()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
            // SAFETY: caller supplies newly allocated sets without GPU users. The
            // images outlive all submissions; binding 5 is a sampled depth image.
            unsafe {
                self.ctx.device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&info)],
                    &[],
                );
            }
        }
    }
    pub fn record(
        &self,
        command: vk::CommandBuffer,
        frame: usize,
        world: &RenderWorld,
        meshes: &HashMap<MeshHandle, GpuMesh>,
        viewport: govno_render::RenderExtent,
    ) -> Result<(), BoxError> {
        let matrix = world
            .directional_shadow
            .map(|s| s.cascades(&world.camera, viewport, world.light_direction))
            .transpose()?;
        let image = &self.images[frame];
        let device = &self.ctx.device;
        // SAFETY: caller has begun command recording after the frame fence wait.
        // Per-frame depth image and meshes remain live through submission completion.
        unsafe {
            let barrier = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(depth_range())];
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
            let depth = vk::RenderingAttachmentInfo::default()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            let extent = vk::Extent2D {
                width: SHADOW_SIZE * 2,
                height: SHADOW_SIZE * 2,
            };
            let rect = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            };
            device.cmd_begin_rendering(
                command,
                &vk::RenderingInfo::default()
                    .render_area(rect)
                    .layer_count(1)
                    .depth_attachment(&depth),
            );
            device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            for cascade in 0..4 {
                device.cmd_set_viewport(
                    command,
                    0,
                    &[vk::Viewport {
                        x: (cascade % 2) as f32 * SHADOW_SIZE as f32,
                        y: (cascade / 2) as f32 * SHADOW_SIZE as f32,
                        width: SHADOW_SIZE as f32,
                        height: SHADOW_SIZE as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                device.cmd_set_scissor(
                    command,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D {
                            x: (cascade % 2) as i32 * SHADOW_SIZE as i32,
                            y: (cascade / 2) as i32 * SHADOW_SIZE as i32,
                        },
                        extent: vk::Extent2D {
                            width: SHADOW_SIZE,
                            height: SHADOW_SIZE,
                        },
                    }],
                );
                if let Some(cascades) = &matrix {
                    let matrix = cascades.matrices[cascade];
                    for instance in &world.instances {
                        let mesh = meshes.get(&instance.mesh).ok_or("unknown shadow mesh")?;
                        let matrix = (matrix * instance.transform.matrix()).to_cols_array();
                        let mut bytes = [0; 64];
                        for (destination, value) in
                            bytes.as_chunks_mut::<4>().0.iter_mut().zip(matrix)
                        {
                            destination.copy_from_slice(&value.to_ne_bytes());
                        }
                        device.cmd_push_constants(
                            command,
                            self.pipeline.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            &bytes,
                        );
                        device.cmd_bind_vertex_buffers(command, 0, &[mesh.vertices.handle], &[0]);
                        device.cmd_bind_index_buffer(
                            command,
                            mesh.indices.handle,
                            0,
                            vk::IndexType::UINT32,
                        );
                        device.cmd_draw_indexed(command, mesh.index_count, 1, 0, 0, 0);
                    }
                }
            }
            device.cmd_end_rendering(command);
            let ready = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(depth_range())];
            device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&ready),
            );
        }
        Ok(())
    }
}

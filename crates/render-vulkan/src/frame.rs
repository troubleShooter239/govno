use crate::{context::DeviceContext, mesh::GpuMesh, swapchain::Swapchain};
use ash::vk;
use govno_core::BoxError;
use govno_render::{MeshHandle, RenderWorld};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone, Copy, Default)]
pub struct DrawCounts {
    pub indirect_calls: u32,
    pub descriptor_binds: u32,
    // With GPU culling, visibility counts describe the latest completed frame.
    pub submitted: u32,
    pub culled: u32,
    pub shadow: u32,
}
pub(crate) struct Frame {
    pub ctx: Arc<DeviceContext>,
    pub pool: vk::CommandPool,
    pub command: vk::CommandBuffer,
    pub acquired: vk::Semaphore,
    pub serial: u64,
    pub query_pool: vk::QueryPool,
    measured_serial: u64,
    timing_enabled: bool,
}
impl Frame {
    pub fn new(ctx: Arc<DeviceContext>) -> Result<Self, BoxError> {
        let mut frame = Self {
            ctx,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            acquired: vk::Semaphore::null(),
            serial: 0,
            query_pool: vk::QueryPool::null(),
            measured_serial: 0,
            timing_enabled: false,
        };
        // SAFETY: every created resource is recorded before further fallible calls.
        unsafe {
            frame.query_pool = frame.ctx.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(6),
                None,
            )?;
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
        }
        Ok(frame)
    }
    pub fn enable_timing(&mut self) {
        self.measured_serial = self.serial;
        self.timing_enabled = true;
    }
    /// Call after the slot's timeline serial completed, before resetting queries.
    pub fn timing(&mut self) -> Result<Option<[f64; 6]>, BoxError> {
        if !self.timing_enabled || self.serial == 0 || self.serial == self.measured_serial {
            return Ok(None);
        }
        let mut values = [0_u64; 6];
        // SAFETY: caller waited for this slot's submission, and all six queries were written.
        unsafe {
            self.ctx.device.get_query_pool_results(
                self.query_pool,
                0,
                &mut values,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let bits = self.ctx.capabilities.timestamp_valid_bits;
        let mask = if bits >= 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        };
        let scale = f64::from(self.ctx.capabilities.timestamp_period) * 1e-6;
        let interval =
            |a: usize, b: usize| ((values[b].wrapping_sub(values[a])) & mask) as f64 * scale;
        self.measured_serial = self.serial;
        Ok(Some([
            interval(0, 4),
            interval(1, 2),
            interval(5, 3),
            interval(3, 4),
            interval(0, 1),
            interval(2, 5),
        ]))
    }
    pub fn record(
        &self,
        chain: &Swapchain,
        indices: (usize, usize),
        scene: (
            &crate::scene_gpu::SceneGpu,
            Option<&crate::culling::Culling>,
        ),
        world: &RenderWorld,
        meshes: &HashMap<MeshHandle, GpuMesh>,
        mut ui: Option<&mut crate::ui::Ui>,
    ) -> Result<DrawCounts, BoxError> {
        let (image_index, frame_index) = indices;
        let (scene, culling) = scene;
        let device = &self.ctx.device;
        let mut counts = DrawCounts::default();
        let frustum = govno_render::Frustum::from_view_projection(world.camera.view_projection(
            govno_render::RenderExtent {
                width: chain.extent.width,
                height: chain.extent.height,
            },
        )?)?;
        let image = chain.images[image_index];
        let depth = &chain.depth[frame_index];
        let hdr = &chain.post.targets[frame_index];
        let extent = chain.extent;
        // SAFETY: frame timeline value has completed, so its command pool, uniform and depth
        // target are no longer in flight. Meshes are validated and retained through
        // the submission serial. Swapchain image has been acquired.
        unsafe {
            device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            if self.timing_enabled {
                device.cmd_reset_query_pool(self.command, self.query_pool, 0, 6);
            }
            if self.timing_enabled {
                device.cmd_write_timestamp2(
                    self.command,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    self.query_pool,
                    0,
                );
            }
            if self.timing_enabled && culling.is_none() {
                device.cmd_write_timestamp2(
                    self.command,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    self.query_pool,
                    1,
                );
            }
            let effects = chain.post.effects.as_ref().ok_or("post effects missing")?;
            use crate::graph::{Binding, RESOURCE_COUNT, Resource};
            let mut bindings = [Binding::Image {
                handle: image,
                range: crate::swapchain::color_range(),
            }; RESOURCE_COUNT];
            bindings[Resource::Output as usize] = Binding::Image {
                handle: image,
                range: crate::swapchain::color_range(),
            };
            bindings[Resource::Hdr as usize] = Binding::image(hdr, crate::swapchain::color_range());
            bindings[Resource::Depth as usize] =
                Binding::image(depth, crate::memory::depth_range());
            bindings[Resource::Shadow as usize] = Binding::image(
                &scene.shadow().images[frame_index],
                crate::memory::depth_range(),
            );
            bindings[Resource::Bloom as usize] =
                Binding::image(&effects.bloom[frame_index], crate::swapchain::color_range());
            bindings[Resource::Scratch as usize] = Binding::image(
                &effects.scratch[frame_index],
                crate::swapchain::color_range(),
            );
            bindings[Resource::Exposure as usize] = Binding::Buffer {
                handle: effects.exposure.handle,
                size: 16,
            };
            bindings[Resource::Lighting as usize] = scene.lighting.binding(frame_index);
            bindings[Resource::Objects as usize] = scene.objects_binding(frame_index);
            if let Some(culling) = culling {
                let [input, commands, counts, readback] = culling.bindings(frame_index);
                bindings[Resource::CullInput as usize] = input;
                bindings[Resource::Commands as usize] = commands;
                bindings[Resource::Counts as usize] = counts;
                bindings[Resource::Readback as usize] = readback;
            }
            for pass in &chain.graph.plan.passes {
                crate::graph::barriers(&self.ctx, self.command, pass, &bindings);
                match pass.tag {
                    crate::graph::Pass::HostLighting => {}
                    crate::graph::Pass::LightCull => {
                        scene.lighting.record(self.command, frame_index);
                        if self.timing_enabled {
                            device.cmd_write_timestamp2(
                                self.command,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                self.query_pool,
                                5,
                            );
                        }
                    }
                    crate::graph::Pass::ResetCull => culling
                        .ok_or("missing culling resources")?
                        .reset(self.command, frame_index),
                    crate::graph::Pass::Cull => {
                        culling
                            .ok_or("missing culling resources")?
                            .dispatch(self.command, frame_index);
                        if self.timing_enabled {
                            device.cmd_write_timestamp2(
                                self.command,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                self.query_pool,
                                1,
                            );
                        }
                    }
                    crate::graph::Pass::ReadbackCull => culling
                        .ok_or("missing culling resources")?
                        .copy_counts(self.command, frame_index),
                    crate::graph::Pass::HostReadback => {}
                    crate::graph::Pass::Shadow => {
                        counts.shadow = scene.shadow().record(
                            self.command,
                            frame_index,
                            world,
                            meshes,
                            govno_render::RenderExtent {
                                width: extent.width,
                                height: extent.height,
                            },
                            scene.sets[frame_index],
                        )?;
                        if self.timing_enabled {
                            device.cmd_write_timestamp2(
                                self.command,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                self.query_pool,
                                2,
                            );
                            if !self.ctx.forward_plus {
                                device.cmd_write_timestamp2(
                                    self.command,
                                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                    self.query_pool,
                                    5,
                                );
                            }
                        }
                    }
                    crate::graph::Pass::Forward => {
                        let colors = [vk::RenderingAttachmentInfo::default()
                            .image_view(hdr.view)
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
                        if let Some(sets) = scene.global_sets(frame_index) {
                            counts.descriptor_binds += 1;
                            device.cmd_bind_descriptor_sets(
                                self.command,
                                vk::PipelineBindPoint::GRAPHICS,
                                chain.pipeline.layout,
                                0,
                                &sets,
                                &[],
                            );
                        }
                        if let Some(culling) = culling {
                            let (commands, batch_counts) = culling.buffers(frame_index);
                            for (id, batch) in culling.batches(frame_index).iter().enumerate() {
                                let mesh = meshes.get(&batch.mesh).ok_or("stale batch mesh")?;
                                if scene.texture_layout().is_none() {
                                    counts.descriptor_binds += 1;
                                    device.cmd_bind_descriptor_sets(
                                        self.command,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        chain.pipeline.layout,
                                        0,
                                        &scene.descriptors(frame_index, &batch.material)?,
                                        &[],
                                    );
                                }
                                device.cmd_bind_vertex_buffers(
                                    self.command,
                                    0,
                                    &[mesh.vertices.handle],
                                    &[0],
                                );
                                device.cmd_bind_index_buffer(
                                    self.command,
                                    mesh.indices.handle,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                device.cmd_draw_indexed_indirect_count(
                                    self.command,
                                    commands,
                                    u64::from(batch.base) * crate::culling::COMMAND_BYTES,
                                    batch_counts,
                                    id as u64 * 4,
                                    batch.capacity,
                                    crate::culling::COMMAND_BYTES as u32,
                                );
                                counts.indirect_calls += 1;
                            }
                        } else {
                            for (object_id, instance) in world.instances.iter().enumerate() {
                                let mesh = meshes
                                    .get(&instance.mesh)
                                    .ok_or("stale or foreign mesh handle")?;
                                if world.frustum_culling
                                    && mesh
                                        .bounds
                                        .transformed(instance.transform.matrix())
                                        .is_some_and(|bounds| !frustum.intersects(bounds))
                                {
                                    counts.culled += 1;
                                    continue;
                                }
                                counts.submitted += 1;
                                if scene.texture_layout().is_none() {
                                    counts.descriptor_binds += 1;
                                    device.cmd_bind_descriptor_sets(
                                        self.command,
                                        vk::PipelineBindPoint::GRAPHICS,
                                        chain.pipeline.layout,
                                        0,
                                        &scene.descriptors(frame_index, &instance.material)?,
                                        &[],
                                    );
                                }

                                device.cmd_bind_vertex_buffers(
                                    self.command,
                                    0,
                                    &[mesh.vertices.handle],
                                    &[0],
                                );
                                device.cmd_bind_index_buffer(
                                    self.command,
                                    mesh.indices.handle,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                device.cmd_draw_indexed(
                                    self.command,
                                    mesh.index_count,
                                    1,
                                    0,
                                    0,
                                    object_id as u32,
                                );
                            }
                        }
                        device.cmd_end_rendering(self.command);
                        if self.timing_enabled {
                            device.cmd_write_timestamp2(
                                self.command,
                                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                                self.query_pool,
                                3,
                            );
                        }
                    }
                    crate::graph::Pass::Effects(effect) => {
                        chain
                            .post
                            .effects
                            .as_ref()
                            .ok_or("post effects missing")?
                            .record(effect, self.command, frame_index, world.post_process);
                    }
                    crate::graph::Pass::Composite => {
                        let output = [vk::RenderingAttachmentInfo::default()
                            .image_view(chain.views[image_index])
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .store_op(vk::AttachmentStoreOp::STORE)];
                        device.cmd_begin_rendering(
                            self.command,
                            &vk::RenderingInfo::default()
                                .render_area(vk::Rect2D {
                                    offset: Default::default(),
                                    extent,
                                })
                                .layer_count(1)
                                .color_attachments(&output),
                        );
                        let post = chain
                            .post
                            .pipeline
                            .as_ref()
                            .ok_or("post pipeline not initialized")?;
                        device.cmd_bind_pipeline(
                            self.command,
                            vk::PipelineBindPoint::GRAPHICS,
                            post.pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            self.command,
                            vk::PipelineBindPoint::GRAPHICS,
                            post.layout,
                            0,
                            &[chain.post.sets[frame_index]],
                            &[],
                        );
                        let output = crate::post::output_constants(
                            world.post_process,
                            chain.srgb_attachment,
                        );
                        device.cmd_push_constants(
                            self.command,
                            post.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            &output,
                        );
                        device.cmd_draw(self.command, 3, 1, 0, 0);
                        if let Some(ui) = ui.as_deref_mut() {
                            ui.record(frame_index, self.command, extent)?;
                        }
                        device.cmd_end_rendering(self.command);
                    }
                    crate::graph::Pass::Present => {}
                }
            }
            if self.timing_enabled {
                device.cmd_write_timestamp2(
                    self.command,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    self.query_pool,
                    4,
                );
            }
            device.end_command_buffer(self.command)?;
        }
        Ok(counts)
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        // SAFETY: renderer waits idle before dropping frames; construction failures
        // have no submitted work. Null handles are permitted for destruction.
        unsafe {
            self.ctx.device.destroy_query_pool(self.query_pool, None);
            self.ctx.device.destroy_semaphore(self.acquired, None);
            self.ctx.device.destroy_command_pool(self.pool, None);
        }
    }
}

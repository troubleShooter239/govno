use crate::context::DeviceContext;
use ash::vk;
use govno_core::BoxError;
use std::sync::Arc;
type Painter = egui_ash_renderer::Renderer<egui_ash_renderer::allocator::DefaultAllocator>;

pub(crate) struct Ui {
    // Painters must be destroyed while the retained device still exists.
    painters: Vec<Painter>,
    ctx: Arc<DeviceContext>,
    format: vk::Format,
    primitives: Vec<egui::ClippedPrimitive>,
    pixels_per_point: f32,
    pending_free: Vec<egui::TextureId>,
}
impl Ui {
    pub fn new(
        ctx: Arc<DeviceContext>,
        format: vk::Format,
        frames: usize,
    ) -> Result<Self, BoxError> {
        let mut painters = Vec::new();
        for _ in 0..frames {
            painters.push(Painter::with_default_allocator(
                ctx.instance(),
                ctx.physical,
                ctx.device.clone(),
                mode(format),
                egui_ash_renderer::Options {
                    in_flight_frames: 1,
                    srgb_framebuffer: matches!(
                        format,
                        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                    ),
                    ..Default::default()
                },
            )?);
        }
        Ok(Self {
            painters,
            ctx,
            format,
            primitives: Vec::new(),
            pixels_per_point: 1.0,
            pending_free: Vec::new(),
        })
    }
    /// Caller has waited for all GPU users before updating/releasing atlas images.
    pub fn update(
        &mut self,
        context: &egui::Context,
        mut output: egui::FullOutput,
        pool: vk::CommandPool,
    ) -> Result<(), BoxError> {
        for id in self.pending_free.drain(..) {
            for painter in &mut self.painters {
                painter.free_texture(id)?;
            }
        }
        for (id, deltas) in output.textures_delta.set.drain() {
            for delta in deltas {
                for painter in &mut self.painters {
                    painter.set_texture(self.ctx.queue, pool, id, &delta)?;
                }
            }
        }
        self.pending_free.extend(output.textures_delta.free.drain());
        self.pixels_per_point = output.pixels_per_point;
        self.primitives = context.tessellate(output.shapes, output.pixels_per_point);
        Ok(())
    }
    pub fn needs_wait(&self, output: &egui::FullOutput) -> bool {
        !self.pending_free.is_empty() || !output.textures_delta.set.is_empty()
    }
    pub fn format(&mut self, format: vk::Format) -> Result<(), BoxError> {
        if self.format != format {
            for painter in &mut self.painters {
                painter.set_render_mode(mode(format))?;
            }
            self.format = format;
        }
        Ok(())
    }
    /// Frame completion protects this painter's vertex/index buffers, including hidden UI frames.
    pub fn record(
        &mut self,
        index: usize,
        command: vk::CommandBuffer,
        extent: vk::Extent2D,
    ) -> Result<(), BoxError> {
        self.painters[index].cmd_draw(command, extent, self.pixels_per_point, &self.primitives)?;
        Ok(())
    }
}
fn mode(format: vk::Format) -> egui_ash_renderer::RenderMode {
    egui_ash_renderer::RenderMode::DynamicRendering(egui_ash_renderer::DynamicRendering {
        color_attachment_format: format,
        depth_attachment_format: None,
        stencil_attachment_format: None,
    })
}

use govno_platform::{NativeWindowEvent, Window};
use std::{sync::Arc, time::Duration};

pub struct DebugUi {
    pub context: egui::Context,
    state: egui_winit::State,
    window: Arc<Window>,
    visible: bool,
}
impl DebugUi {
    pub fn new(window: Arc<Window>) -> Self {
        let context = egui::Context::default();
        let state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        Self {
            context,
            state,
            window,
            visible: true,
        }
    }
    pub fn event(
        &mut self,
        window: &Window,
        event: &NativeWindowEvent,
        simulation: &mut govno_ecs::Simulation,
    ) -> bool {
        if let NativeWindowEvent::KeyboardInput { event, .. } = event
            && event.physical_key == govno_platform::PhysicalKey::Code(govno_platform::KeyCode::F1)
            && event.state.is_pressed()
            && !event.repeat
        {
            self.visible = !self.visible;
            simulation.input(govno_input::InputEvent::FocusLost);
        }
        let response = self.state.on_window_event(window, event);
        if response.consumed {
            simulation.input(govno_input::InputEvent::FocusLost);
        }
        response.consumed
    }
    pub fn draw(
        &mut self,
        simulation: &mut govno_ecs::Simulation,
        world: &mut govno_render::RenderWorld,
        elapsed: Duration,
        cached: usize,
    ) -> egui::FullOutput {
        let input = self.state.take_egui_input(&self.window);
        let output = self.context.run_ui(input, |ui| {
            if !self.visible {
                return;
            }
            egui::Window::new("Sandbox | F1")
                .default_width(280.0)
                .show(ui.ctx(), |ui| {
                    ui.label(format!("Frame: {:.2} ms", elapsed.as_secs_f64() * 1000.0));
                    ui.label(format!(
                        "Entities: {} | Bodies: {}",
                        simulation.entity_count(),
                        simulation.body_count()
                    ));
                    ui.label(format!(
                        "Tick: {} | Cached models: {}",
                        simulation.ticks, cached
                    ));
                    ui.label(format!(
                        "Discarded: {:.3} s",
                        simulation.discarded.as_secs_f64()
                    ));
                    ui.checkbox(&mut simulation.paused, "Pause simulation");
                    ui.add(
                        egui::Slider::new(&mut world.post_process.exposure_ev, -5.0..=5.0)
                            .text("Exposure EV"),
                    );
                    ui.checkbox(&mut world.frustum_culling, "Frustum culling");
                    ui.separator();
                    ui.label("WASD move | RMB look | Shift sprint");
                    ui.label("Space jump | F push | R respawn | F5 reload");
                });
        });
        self.state
            .handle_platform_output(&self.window, output.platform_output.clone());
        output
    }
}

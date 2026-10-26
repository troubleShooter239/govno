//! Desktop window/event-loop ownership. No dependency on graphics or ECS.
mod input;
use govno_core::BoxError;
pub use input::{InputEvent, InputKey};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
pub use winit::window::Window;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, KeyCode, NamedKey, PhysicalKey},
    window::WindowId,
};

pub trait WindowDriver {
    fn input(&mut self, _event: InputEvent) {}
    fn resize(&mut self, width: u32, height: u32);
    /// Returns true only for a presented frame.
    fn draw(&mut self) -> Result<bool, BoxError>;
    fn shutdown(&mut self) -> Result<(), BoxError>;
}
#[derive(Default)]
pub struct RunOptions {
    pub frame_limit: Option<u64>,
    /// Automated desktop resize/minimize/restore exercise.
    pub smoke: bool,
}
type Factory = Box<dyn FnOnce(Arc<Window>) -> Result<Box<dyn WindowDriver>, BoxError>>;
struct DesktopApp {
    factory: Option<Factory>,
    driver: Option<Box<dyn WindowDriver>>,
    window: Option<Arc<Window>>,
    error: Option<BoxError>,
    options: RunOptions,
    frames: u64,
    minimized: bool,
    look_active: bool,
    restore_at: Option<Instant>,
    deadline: Instant,
}
impl DesktopApp {
    fn fail(&mut self, event_loop: &ActiveEventLoop, error: BoxError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
        event_loop.exit();
    }
    fn release_cursor(&mut self) {
        self.look_active = false;
        if let Some(window) = &self.window {
            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
    }
    fn stop(&mut self) {
        self.release_cursor();
        if let Some(mut driver) = self.driver.take() {
            if let Err(error) = driver.shutdown() {
                if self.error.is_none() {
                    self.error = Some(error);
                } else {
                    eprintln!("additional shutdown error: {error}");
                }
            }
            drop(driver);
        }
        self.window.take();
    }
}
impl ApplicationHandler for DesktopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let Some(factory) = self.factory.take() else {
            return;
        };
        let result = (|| {
            let window = Arc::new(
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("GOVNO | Basic 3D | WASD + RMB look | Q/E vertical")
                        .with_inner_size(PhysicalSize::new(1280, 720)),
                )?,
            );
            let driver = factory(window.clone())?;
            self.driver = Some(driver);
            self.window = Some(window);
            Ok::<_, BoxError>(())
        })();
        if let Err(error) = result {
            self.fail(event_loop, error);
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if self.window.as_ref().is_none_or(|w| w.id() != id) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                event_loop.exit()
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let key = match code {
                        KeyCode::KeyW => Some(InputKey::W),
                        KeyCode::KeyA => Some(InputKey::A),
                        KeyCode::KeyS => Some(InputKey::S),
                        KeyCode::KeyD => Some(InputKey::D),
                        KeyCode::KeyQ => Some(InputKey::Q),
                        KeyCode::KeyE => Some(InputKey::E),
                        KeyCode::ShiftLeft => Some(InputKey::Shift),
                        _ => None,
                    };
                    if let Some(key) = key
                        && let Some(driver) = &mut self.driver
                    {
                        driver.input(InputEvent::Key {
                            key,
                            pressed: event.state == ElementState::Pressed,
                        });
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Right,
                ..
            } => {
                if state == ElementState::Pressed {
                    if let Some(window) = &self.window {
                        let captured = window
                            .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                            .or_else(|_| {
                                window.set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            })
                            .is_ok();
                        self.look_active = captured;
                        window.set_cursor_visible(!captured);
                    }
                } else {
                    self.release_cursor();
                }
            }
            WindowEvent::Focused(false) => {
                self.release_cursor();
                if let Some(driver) = &mut self.driver {
                    driver.input(InputEvent::FocusLost);
                }
            }
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if let Some(driver) = &mut self.driver {
                    driver.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if self.minimized || self.restore_at.is_some() {
                    return;
                }
                let Some(driver) = &mut self.driver else {
                    return;
                };
                match driver.draw() {
                    Ok(true) => {
                        self.frames += 1;
                        if self
                            .options
                            .frame_limit
                            .is_some_and(|limit| self.frames >= limit)
                        {
                            event_loop.exit();
                        } else if self.options.smoke {
                            let window = self.window.as_ref().expect("live window");
                            match self.frames {
                                10 => {
                                    let _ = window.request_inner_size(PhysicalSize::new(800, 600));
                                }
                                30 => {
                                    let _ = window.request_inner_size(PhysicalSize::new(1440, 810));
                                }
                                50 => {
                                    // Force a zero extent even if the OS does not emit one on minimize.
                                    driver.resize(0, 0);
                                    match driver.draw() {
                                        Ok(false) => {}
                                        Ok(true) => {
                                            self.fail(
                                                event_loop,
                                                "zero extent rendered unexpectedly".into(),
                                            );
                                            return;
                                        }
                                        Err(error) => {
                                            self.fail(event_loop, error);
                                            return;
                                        }
                                    }
                                    window.set_minimized(true);
                                    self.restore_at =
                                        Some(Instant::now() + Duration::from_millis(300));
                                }
                                60 => {
                                    driver.input(InputEvent::Key {
                                        key: InputKey::W,
                                        pressed: true,
                                    });
                                    driver.input(InputEvent::Look { x: 24.0, y: -8.0 });
                                }
                                70 => driver.input(InputEvent::FocusLost),
                                _ => {}
                            }
                        }
                    }
                    Ok(false) => {}
                    Err(error) => self.fail(event_loop, error),
                }
            }
            _ => {}
        }
    }
    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if self.look_active
            && let DeviceEvent::MouseMotion { delta: (x, y) } = event
            && let Some(driver) = &mut self.driver
        {
            driver.input(InputEvent::Look {
                x: x as f32,
                y: y as f32,
            });
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.options.frame_limit.is_some() && Instant::now() > self.deadline {
            self.fail(
                event_loop,
                "frame-limited run timed out after 30 seconds".into(),
            );
            return;
        }
        if let Some(restore_at) = self.restore_at {
            if Instant::now() >= restore_at {
                if let Some(window) = &self.window {
                    window.set_minimized(false);
                    let size = window.inner_size();
                    if let Some(driver) = &mut self.driver {
                        driver.resize(size.width, size.height);
                    }
                    self.minimized = size.width == 0 || size.height == 0;
                }
                self.restore_at = None;
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(restore_at));
                return;
            }
        }
        // Timer also lets finite smoke runs time out if the window stays minimized.
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(16),
        ));
        if !self.minimized
            && let Some(window) = &self.window
        {
            window.request_redraw();
        }
    }
    fn exiting(&mut self, _: &ActiveEventLoop) {
        self.stop();
    }
}
pub fn run(
    options: RunOptions,
    factory: impl FnOnce(Arc<Window>) -> Result<Box<dyn WindowDriver>, BoxError> + 'static,
) -> Result<u64, BoxError> {
    if options.frame_limit == Some(0) {
        return Err("frame limit must be positive".into());
    }
    let event_loop = EventLoop::new()?;
    let mut app = DesktopApp {
        factory: Some(Box::new(factory)),
        driver: None,
        window: None,
        error: None,
        options,
        frames: 0,
        minimized: false,
        look_active: false,
        restore_at: None,
        deadline: Instant::now() + Duration::from_secs(30),
    };
    let loop_result = event_loop.run_app(&mut app);
    app.stop();
    loop_result?;
    if let Some(error) = app.error {
        return Err(error);
    }
    Ok(app.frames)
}

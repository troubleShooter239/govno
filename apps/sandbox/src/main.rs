mod scene;
use glam::{Quat, Vec3};
use govno_core::BoxError;
use govno_platform::{InputEvent, InputKey, RunOptions, WindowDriver};
use govno_render::{FrameStatus, MeshHandle, RenderExtent, RenderWorld};
use govno_render_vulkan::{ValidationStats, VulkanRenderer};
use govno_runtime::Engine;
use std::{collections::HashSet, sync::Arc, time::Instant};

struct GraphicsApp {
    renderer: VulkanRenderer,
    world: RenderWorld,
    keys: HashSet<InputKey>,
    last_frame: Instant,
    elapsed: f32,
    frames: u64,
    animated: usize,
    temporary: Option<MeshHandle>,
    smoke: bool,
}
impl WindowDriver for GraphicsApp {
    fn input(&mut self, event: InputEvent) {
        match event {
            InputEvent::Key { key, pressed } => {
                if pressed {
                    self.keys.insert(key);
                } else {
                    self.keys.remove(&key);
                }
            }
            InputEvent::Look { x, y } => self.world.camera.look(x, y),
            InputEvent::FocusLost => self.keys.clear(),
        }
    }
    fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(RenderExtent { width, height });
    }
    fn draw(&mut self) -> Result<bool, BoxError> {
        let now = Instant::now();
        let seconds = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        self.elapsed += seconds;
        let held = |key| if self.keys.contains(&key) { 1.0 } else { 0.0 };
        self.world.camera.move_local(
            Vec3::new(
                held(InputKey::D) - held(InputKey::A),
                held(InputKey::E) - held(InputKey::Q),
                held(InputKey::W) - held(InputKey::S),
            ),
            seconds,
            if self.keys.contains(&InputKey::Shift) {
                10.0
            } else {
                4.0
            },
        );
        self.world.instances[self.animated].transform.rotation =
            Quat::from_rotation_y(self.elapsed * 0.4);
        if self.frames == 20
            && let Some(handle) = self.temporary.take()
        {
            self.world
                .instances
                .retain(|instance| instance.mesh != handle);
            self.renderer.remove_mesh(handle)?;
            tracing::info!("smoke: mesh retired after submitted draws");
        }
        if self.smoke && self.frames == 40 {
            self.renderer
                .set_base_color_texture(&govno_render::TextureData {
                    width: 2,
                    height: 2,
                    rgba8: vec![
                        255, 255, 255, 255, 155, 155, 155, 255, 155, 155, 155, 255, 255, 255, 255,
                        255,
                    ],
                })?;
            tracing::info!("smoke: texture replaced after submitted frames");
        }
        let presented = self.renderer.draw(&self.world)? == FrameStatus::Presented;
        if presented {
            self.frames += 1;
        }
        Ok(presented)
    }
    fn shutdown(&mut self) -> Result<(), BoxError> {
        tracing::info!(position=?self.world.camera.position, yaw=self.world.camera.yaw, "camera shutdown state");
        self.renderer.wait_idle()
    }
}
fn main() -> Result<(), BoxError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init()?;
    let mut options = RunOptions::default();
    let mut validation = cfg!(debug_assertions);
    let mut gltf_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => {
                let value: u64 = args
                    .next()
                    .ok_or("--frames requires a positive integer")?
                    .parse()?;
                if value == 0 {
                    return Err("--frames must be greater than zero".into());
                }
                options.frame_limit = Some(value);
            }
            "--gltf" => gltf_path = Some(args.next().ok_or("--gltf requires a path")?),
            "--smoke" => options.smoke = true,
            "--validation" => validation = true,
            "--no-validation" => validation = false,
            "--help" => {
                println!(
                    "govno-sandbox [--gltf PATH] [--frames N] [--smoke] [--validation|--no-validation]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if options.smoke && options.frame_limit.is_none_or(|n| n < 80) {
        return Err("--smoke requires --frames N with N >= 80".into());
    }
    let imported = gltf_path.map(govno_assets::load_gltf).transpose()?;
    let smoke = options.smoke;
    let stats = Arc::new(ValidationStats::default());
    let callback_stats = stats.clone();
    let result = Engine::builder().build()?.run(|| {
        let frames = govno_platform::run(options, move |window| {
            let size = window.inner_size();
            let mut renderer = VulkanRenderer::new(
                window,
                RenderExtent {
                    width: size.width,
                    height: size.height,
                },
                validation,
                callback_stats,
            )?;
            let (mut world, animated, temporary) = scene::build(&mut renderer, smoke)?;
            if let Some(primitives) = imported {
                let white = renderer.upload_texture(&govno_render::TextureData {
                    width: 1,
                    height: 1,
                    rgba8: vec![255; 4],
                })?;
                for primitive in primitives {
                    let mesh = renderer.upload_mesh(&primitive.mesh)?;
                    let mut material = primitive.material;
                    material.base_color_texture = Some(match primitive.base_color_texture {
                        Some(data) => {
                            renderer.upload_texture_with_sampling(&data, primitive.sampling)?
                        }
                        None => white,
                    });
                    world.instances.push(govno_render::DrawInstance {
                        mesh,
                        material,
                        transform: Default::default(),
                    });
                }
            }
            Ok(Box::new(GraphicsApp {
                renderer,
                world,
                keys: HashSet::new(),
                last_frame: Instant::now(),
                elapsed: 0.0,
                frames: 0,
                animated,
                temporary,
                smoke,
            }))
        })?;
        tracing::info!(frames, "desktop run completed");
        Ok(())
    });
    // Read after teardown, so destruction-time validation errors count as well.
    tracing::info!(
        errors = stats.errors(),
        warnings = stats.warnings(),
        "validation summary"
    );
    result?;
    if stats.errors() > 0 {
        return Err("Vulkan validation errors were reported".into());
    }
    Ok(())
}

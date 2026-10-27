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
    shadows: bool,
    environment: bool,
    specular: bool,
    environment_data: govno_render::EnvironmentRadiance,
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
        self.world.post_process.delta_seconds = seconds;
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
                    color_space: Default::default(),
                    width: 2,
                    height: 2,
                    rgba8: vec![
                        255, 255, 255, 255, 155, 155, 155, 255, 155, 155, 155, 255, 255, 255, 255,
                        255,
                    ],
                })?;
            tracing::info!("smoke: texture replaced after submitted frames");
        }
        if self.smoke && self.environment {
            if self.frames == 26 {
                self.world.diffuse_environment = None;
                self.world.specular_environment = false;
                tracing::info!("smoke: diffuse environment disabled");
            }
            if self.frames == 28 {
                let environment =
                    govno_render::EnvironmentRadiance::constant(Vec3::new(0.05, 0.1, 0.2))?;
                self.renderer.set_environment_gpu(&environment, 0)?;
                self.world.diffuse_environment =
                    Some(govno_render::DiffuseEnvironment::constant(Vec3::ZERO)?);
                self.world.specular_environment = self.specular;
                tracing::info!("smoke: constant HDR environment enabled");
            }
            if self.frames == 32 {
                self.renderer
                    .set_environment_gpu(&self.environment_data, 0)?;
                self.world.diffuse_environment =
                    Some(govno_render::DiffuseEnvironment::constant(Vec3::ZERO)?);
                self.world.specular_environment = self.specular;
                tracing::info!("smoke: projected panorama environment restored");
            }
        }
        if self.smoke && self.shadows {
            if self.frames == 22 {
                self.world.directional_shadow = None;
                tracing::info!("smoke: shadows disabled");
            }
            if self.frames == 25 {
                self.world.directional_shadow = Some(Default::default());
                tracing::info!("smoke: shadows enabled");
            }
        }
        if self.smoke && self.frames == 45 {
            self.world.point_lights[0].position.x = -1.5;
            self.world.point_lights[1].range = 5.0;
            tracing::info!("smoke: point light position/range updated");
        }
        // Alternate material keys while previous frames may still use their pools.
        if self.smoke && (61..=64).contains(&self.frames) {
            let material = &mut self.world.instances[self.animated].material;
            material.emissive_texture = if self.frames % 2 == 1 {
                material.base_color_texture
            } else {
                None
            };
            tracing::info!(
                frame = self.frames,
                "smoke: material descriptor key changed"
            );
        }
        if self.smoke && self.frames == 77 {
            let mut dark = self.world.point_lights[0];
            dark.intensity = Vec3::ZERO;
            self.world
                .point_lights
                .resize(govno_render::MAX_POINT_LIGHTS, dark);
            tracing::info!("smoke: maximum supported light count submitted");
        }
        if self.smoke && self.frames == 78 {
            self.world.point_lights.clear();
            tracing::info!("smoke: zero point lights submitted");
        }
        if self.smoke && self.frames == 79 {
            self.world.point_lights = scene::point_lights();
        }
        if self.smoke && self.frames == 76 {
            let saved = self.world.point_lights.clone();
            self.world
                .point_lights
                .resize(govno_render::MAX_POINT_LIGHTS + 1, saved[0]);
            let rejected = self.renderer.draw(&self.world).is_err();
            self.world.point_lights = saved;
            if !rejected {
                return Err("smoke: excessive point lights accepted".into());
            }
            tracing::info!("smoke: excessive light count rejected; renderer remains usable");
        }
        if self.smoke {
            if self.frames == 75 {
                let saved = self.world.post_process;
                self.world.post_process.exposure_ev = f32::NAN;
                let rejected = self.renderer.draw(&self.world).is_err();
                self.world.post_process = saved;
                if !rejected {
                    return Err("smoke: invalid exposure was accepted".into());
                }
                tracing::info!("smoke: invalid exposure rejected; renderer remains usable");
            }
            if let Some(offset) = match self.frames {
                15 => Some(-1.0),
                35 => Some(2.0),
                75 => Some(-1.0),
                _ => None,
            } {
                self.world.post_process.exposure_ev =
                    (self.world.post_process.exposure_ev + offset).clamp(-16.0, 16.0);
                tracing::info!(
                    exposure_ev = self.world.post_process.exposure_ev,
                    "smoke: exposure updated"
                );
            }
        }
        if self.smoke {
            match self.frames {
                90 => self.world.camera.position = Vec3::new(-3.0, 2.0, 5.0),
                95 => self.world.camera.position = Vec3::new(3.0, 2.0, 5.0),
                100 => self.world.camera.position.z = 35.0,
                105 => self.world.camera.position.z = 70.0,
                110 => self.world.camera.position = Vec3::new(0.0, 3.3, 10.0),
                _ => {}
            }
        }
        let exposure_before = if self.smoke && self.frames == 80 {
            let state = self.renderer.post_diagnostics()?;
            self.world.light_radiance *= 0.1;
            state
        } else {
            None
        };
        if self.smoke && self.frames == 82 {
            self.world.light_radiance *= 10.0;
        }
        let presented = self.renderer.draw(&self.world)? == FrameStatus::Presented;
        if presented
            && let Some(before) = exposure_before
            && let Some(after) = self.renderer.post_diagnostics()?
        {
            let expected = if let Some(a) = self.world.post_process.auto_exposure {
                let target = (a.key / after[3]).clamp(a.min_ev.exp2(), a.max_ev.exp2());
                (before[1]
                    + (target - before[1])
                        * (1.0 - (-self.world.post_process.delta_seconds * a.speed).exp()))
                .clamp(a.min_ev.exp2(), a.max_ev.exp2())
            } else {
                1.0
            };
            if (expected - after[1]).abs() > 0.001 * expected.max(1.0) {
                return Err("GPU temporal exposure differs from CPU reference".into());
            }
            tracing::info!(
                before = before[1],
                after = after[1],
                expected,
                "GPU temporal exposure verified"
            );
        }
        if presented {
            self.frames += 1;
        }
        Ok(presented)
    }
    fn shutdown(&mut self) -> Result<(), BoxError> {
        tracing::info!(position=?self.world.camera.position, yaw=self.world.camera.yaw, "camera shutdown state");
        if self.smoke
            && let Some(state) = self.renderer.post_diagnostics()?
        {
            tracing::info!(
                exposure = state[0],
                adapted = state[1],
                luminance = state[3],
                "GPU exposure diagnostics"
            );
            if !state.iter().all(|v| v.is_finite()) || state[0] <= 0.0 {
                return Err("invalid GPU exposure state".into());
            }
        }
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
    let mut environment_path = None;
    let mut probe_paths = [None, None];
    let mut local_probes = true;
    let mut emission = None;
    let mut debug_cascades = false;
    let mut shadows = true;
    let mut environment = true;
    let mut specular = true;
    let mut post_process = govno_render::PostProcessSettings::default();
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
            "--exposure" => {
                post_process.exposure_ev = args
                    .next()
                    .ok_or("--exposure requires an EV value")?
                    .parse()?;
                post_process.validate()?;
            }
            "--environment" => {
                environment_path = Some(args.next().ok_or("--environment requires an HDR path")?)
            }
            "--probe-left" => {
                probe_paths[0] = Some(args.next().ok_or("--probe-left requires a path")?)
            }
            "--probe-right" => {
                probe_paths[1] = Some(args.next().ok_or("--probe-right requires a path")?)
            }
            "--debug-cascades" => debug_cascades = true,
            "--no-local-probes" => local_probes = false,
            "--gltf" => gltf_path = Some(args.next().ok_or("--gltf requires a path")?),
            "--emission" => {
                let value: f32 = args.next().ok_or("--emission requires a value")?.parse()?;
                if !value.is_finite() || !(0.0..=100.0).contains(&value) {
                    return Err("emission must be in [0,100]".into());
                }
                emission = Some(value);
            }
            "--bloom-strength" => {
                post_process.bloom.intensity = args
                    .next()
                    .ok_or("--bloom-strength requires a value")?
                    .parse()?
            }
            "--bloom-threshold" => {
                post_process.bloom.threshold = args
                    .next()
                    .ok_or("--bloom-threshold requires a value")?
                    .parse()?
            }
            "--no-bloom" => post_process.bloom.intensity = 0.0,
            "--no-auto-exposure" => post_process.auto_exposure = None,
            "--no-shadows" => shadows = false,
            "--no-environment" => environment = false,
            "--no-specular" => specular = false,
            "--smoke" => options.smoke = true,
            "--validation" => validation = true,
            "--no-validation" => validation = false,
            "--help" => {
                println!(
                    "govno-sandbox [--gltf PATH] [--environment PATH] [--exposure EV] [--no-shadows] [--no-environment] [--no-specular] [--frames N] [--probe-left PATH] [--probe-right PATH] [--no-local-probes] [--debug-cascades] [--no-bloom] [--bloom-strength N] [--bloom-threshold N] [--emission N] [--no-auto-exposure] [--smoke] [--validation|--no-validation]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if options.smoke && options.frame_limit.is_none_or(|n| n < 80) {
        return Err("--smoke requires --frames N with N >= 80".into());
    }
    post_process.validate()?;
    let imported = gltf_path.map(govno_assets::load_gltf).transpose()?;
    let environment_data = if let Some(path) = environment_path {
        let panorama = govno_assets::load_environment(&path)?;
        tracing::info!(path,width=panorama.width,height=panorama.height,exposure=?panorama.exposure,color_correction=?panorama.color_correction,"HDR panorama decoded; stored radiance preserved");
        govno_render::EnvironmentRadiance::new(panorama.width, panorama.height, panorama.radiance)?
    } else {
        scene::environment()?
    };
    let probe_data: Vec<_> = probe_paths
        .into_iter()
        .enumerate()
        .filter_map(|(i, path)| path.map(|path| (i, path)))
        .map(|(i, path)| {
            let panorama = govno_assets::load_environment(path)?;
            Ok::<_, BoxError>((
                i as u32 + 1,
                govno_render::EnvironmentRadiance::new(
                    panorama.width,
                    panorama.height,
                    panorama.radiance,
                )?,
            ))
        })
        .collect::<Result<_, _>>()?;
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
            let (mut world, animated, temporary) =
                scene::build(&mut renderer, smoke, &environment_data)?;
            for (slot, data) in &probe_data {
                renderer.set_environment_gpu(data, *slot)?;
            }
            if smoke {
                renderer.verify_environment(&environment_data)?;
            }
            if let Some(strength) = emission {
                world.instances[animated].material.emissive = Vec3::new(1.0, 0.2, 0.025) * strength;
            }
            if !local_probes {
                world.local_probes.clear();
            }
            if let Some(primitives) = imported {
                let white = renderer.upload_texture(&govno_render::TextureData {
                    color_space: Default::default(),
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
                    if let Some((data, sampling)) = primitive.normal_texture {
                        material.normal_texture =
                            Some(renderer.upload_texture_with_sampling(&data, sampling)?);
                    }
                    if let Some((data, sampling)) = primitive.metallic_roughness_texture {
                        material.metallic_roughness_texture =
                            Some(renderer.upload_texture_with_sampling(&data, sampling)?);
                    }
                    if let Some((data, sampling)) = primitive.emissive_texture {
                        material.emissive_texture =
                            Some(renderer.upload_texture_with_sampling(&data, sampling)?);
                    }
                    if let Some((data, sampling)) = primitive.occlusion_texture {
                        material.occlusion_texture =
                            Some(renderer.upload_texture_with_sampling(&data, sampling)?);
                    }
                    world.instances.push(govno_render::DrawInstance {
                        mesh,
                        material,
                        transform: Default::default(),
                    });
                }
            }
            if let Some(shadow) = &mut world.directional_shadow {
                shadow.debug_cascades = debug_cascades;
            }
            world.post_process = post_process;
            world.specular_environment = environment && specular;
            if !environment {
                world.diffuse_environment = None;
                world.specular_environment = false;
            }
            if !shadows {
                world.directional_shadow = None;
            }
            tracing::info!(exposure_ev = post_process.exposure_ev, "scene exposure");
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
                shadows,
                environment,
                specular,
                environment_data,
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

mod debug_ui;
mod scene;
mod scene_upload;
use glam::Vec3;
use govno_core::BoxError;
use govno_platform::{InputEvent, RunOptions, WindowDriver};
use govno_render::{FrameStatus, RenderExtent, RenderWorld};
use govno_render_vulkan::{ValidationStats, VulkanRenderer};
use govno_runtime::Engine;
use std::{sync::Arc, time::Instant};

struct GraphicsApp {
    renderer: VulkanRenderer,
    world: RenderWorld,
    simulation: govno_ecs::Simulation,
    debug_ui: debug_ui::DebugUi,
    model_assets: govno_assets::AssetManager<Vec<govno_assets::ImportedPrimitive>>,
    last_frame: Instant,
    scene_path: Option<std::path::PathBuf>,
    prepared_scene: Option<govno_scene::PreparedScene>,
    scene_upload: Option<scene_upload::SceneUpload>,
    benchmark_path: Option<std::path::PathBuf>,
    benchmark_warmup: usize,
    expected_offscreen: usize,
}
impl WindowDriver for GraphicsApp {
    fn window_event(
        &mut self,
        window: &govno_platform::Window,
        event: &govno_platform::NativeWindowEvent,
    ) -> bool {
        self.debug_ui.event(window, event, &mut self.simulation)
    }
    fn input(&mut self, event: InputEvent) {
        self.simulation.input(event);
    }
    fn resize(&mut self, width: u32, height: u32) {
        self.renderer.resize(RenderExtent { width, height });
    }
    fn draw(&mut self) -> Result<bool, BoxError> {
        let now = Instant::now();
        let elapsed = now - self.last_frame;
        self.last_frame = now;
        self.world.post_process.delta_seconds = elapsed.as_secs_f32().min(0.1);
        self.simulation.advance(elapsed);
        if self.simulation.take_scene_reload() && self.scene_path.is_some() {
            self.reload_scene()?;
        }
        self.simulation.extract(&mut self.world);
        let output = self.debug_ui.draw(
            &mut self.simulation,
            &mut self.world,
            elapsed,
            self.model_assets.len(),
        );
        self.renderer.update_ui(&self.debug_ui.context, output)?;
        Ok(self.renderer.draw(&self.world)? == FrameStatus::Presented)
    }
    fn shutdown(&mut self) -> Result<(), BoxError> {
        if let Some(path) = &self.benchmark_path {
            let report = self.renderer.benchmark_report(self.benchmark_warmup)?;
            if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, report)?;
            tracing::info!(path=%path.display(), "benchmark report written");
        }
        self.world.instances.clear();
        self.simulation.clear();
        if let Some(upload) = self.scene_upload.take() {
            upload.release(&mut self.renderer)?;
        }
        self.prepared_scene.take();
        self.model_assets.remove_unused();
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
    let mut bindless = true;
    let mut gpu_culling = true;
    let mut verify_culling = false;
    let mut forward_plus = true;
    let mut verify_lighting = false;
    let mut hiz = false;
    let mut gltf_path = None;
    let mut scene_path = None;
    let mut environment_path = None;
    let mut benchmark_path = None;
    let mut benchmark_warmup = 120_usize;
    let mut culling_stress = 0_usize;
    let mut frustum_culling = true;
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
            "--benchmark" => {
                benchmark_path = Some(std::path::PathBuf::from(
                    args.next().ok_or("--benchmark requires a path")?,
                ))
            }
            "--benchmark-warmup" => {
                benchmark_warmup = args.next().ok_or("warmup count required")?.parse()?
            }
            "--culling-stress" => {
                culling_stress = args.next().ok_or("stress count required")?.parse()?;
                if culling_stress > 100000 {
                    return Err("stress count exceeds 100000".into());
                }
            }
            "--no-culling" => frustum_culling = false,
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
            "--scene" => {
                scene_path = Some(
                    std::path::PathBuf::from(args.next().ok_or("--scene requires a path")?)
                        .canonicalize()?,
                )
            }
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
            "--validation" => validation = true,
            "--no-validation" => validation = false,
            "--no-bindless" => bindless = false,
            "--no-gpu-culling" => gpu_culling = false,
            "--verify-culling" => verify_culling = true,
            "--no-forward-plus" => forward_plus = false,
            "--verify-lighting" => verify_lighting = true,
            "--hiz" => hiz = true,
            "--no-hiz" => hiz = false,
            "--help" => {
                println!(
                    "govno-sandbox [--scene PATH | --gltf PATH] [--environment PATH] [--exposure EV] [--no-shadows] [--no-environment] [--no-specular] [--benchmark PATH] [--benchmark-warmup N] [--culling-stress N] [--no-culling] [--frames N] [--probe-left PATH] [--probe-right PATH] [--no-local-probes] [--debug-cascades] [--no-bloom] [--bloom-strength N] [--bloom-threshold N] [--emission N] [--no-auto-exposure] [--validation|--no-validation] [--no-bindless] [--no-gpu-culling] [--verify-culling] [--no-forward-plus] [--verify-lighting] [--hiz|--no-hiz]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if scene_path.is_some() && (gltf_path.is_some() || emission.is_some()) {
        return Err("--scene cannot be combined with demo-only --gltf or --emission".into());
    }
    post_process.validate()?;
    let mut model_assets = govno_assets::AssetManager::default();
    let gltf_import = gltf_path.is_some();
    let prepared_scene = if let Some(path) = &scene_path {
        Some(govno_scene::PreparedScene::load(path, &mut model_assets)?)
    } else if let Some(path) = gltf_path {
        Some(scene_upload::prepare_model(path, &mut model_assets)?)
    } else {
        None
    };
    if culling_stress > 0
        && prepared_scene
            .as_ref()
            .is_some_and(|scene| scene.models().is_empty())
    {
        return Err("--culling-stress needs at least one model".into());
    }
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
    let stats = Arc::new(ValidationStats::default());
    let callback_stats = stats.clone();
    let result = Engine::builder().build()?.run(|| {
        let frames = govno_platform::run(options, move |window| {
            let size = window.inner_size();
            let mut renderer = VulkanRenderer::new(
                window.clone(),
                RenderExtent {
                    width: size.width,
                    height: size.height,
                },
                govno_render_vulkan::RendererOptions {
                    validation,
                    bindless,
                    gpu_culling,
                    verify_culling,
                    forward_plus,
                    verify_lighting,
                    hiz,
                },
                callback_stats,
            )?;
            if benchmark_path.is_some() {
                renderer.start_benchmark();
            }
            let (mut world, animated, mut scene_upload) = if let Some(prepared) = &prepared_scene
                && !gltf_import
            {
                let mut world = scene_upload::world(prepared, Vec::new())?;
                renderer.set_environment_gpu(&environment_data, 0)?;
                let (upload, instances) =
                    scene_upload::SceneUpload::upload(&mut renderer, prepared)?;
                world.instances = instances;
                (world, None, Some(upload))
            } else {
                let (world, animated) = scene::build(&mut renderer, &environment_data)?;
                (world, Some(animated), None)
            };
            world.frustum_culling = frustum_culling;
            add_culling_stress(&mut world, culling_stress)?;
            for (slot, data) in &probe_data {
                renderer.set_environment_gpu(data, *slot)?;
            }
            if let Some(strength) = emission {
                world.instances[animated.ok_or("emission requires the demo scene")?]
                    .material
                    .emissive = Vec3::new(1.0, 0.2, 0.025) * strength;
            }
            if !local_probes {
                world.local_probes.clear();
            }
            if gltf_import && let Some(prepared) = &prepared_scene {
                let (upload, instances) =
                    scene_upload::SceneUpload::upload(&mut renderer, prepared)?;
                world.instances.extend(instances);
                scene_upload = Some(upload);
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
            let debug_ui = debug_ui::DebugUi::new(window.clone());
            let mut simulation = govno_ecs::Simulation::new(world.camera.clone());
            let mut instances = std::mem::take(&mut world.instances);
            if animated.is_none()
                && let Some(prepared) = &prepared_scene
            {
                install_scene(&mut simulation, prepared, &mut instances);
            }
            for (index, instance) in instances.into_iter().enumerate() {
                let half = instance.transform.scale * 0.5;
                let entity = simulation.spawn(instance);
                if Some(index) == animated {
                    simulation.cuboid(entity, half, true);
                } else if animated.is_some_and(|animated| index == animated + 1) {
                    simulation.cuboid(entity, half, false);
                } else if animated.is_some_and(|animated| index < animated) {
                    simulation.sphere(entity, half.x);
                }
            }
            if animated.is_some() {
                simulation.enable_player();
            }
            simulation.extract(&mut world);
            Ok(Box::new(GraphicsApp {
                renderer,
                simulation,
                debug_ui,
                model_assets,
                scene_path,
                prepared_scene,
                scene_upload,
                world,
                last_frame: Instant::now(),
                benchmark_path,
                benchmark_warmup,
                expected_offscreen: culling_stress,
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

fn add_culling_stress(world: &mut RenderWorld, count: usize) -> Result<(), BoxError> {
    if count == 0 {
        return Ok(());
    }
    let source = world
        .instances
        .first()
        .ok_or("culling stress needs at least one draw instance")?;
    let mesh = source.mesh;
    let material = source.material;
    for i in 0..count {
        world.instances.push(govno_render::DrawInstance {
            mesh,
            material,
            transform: govno_render::Transform {
                translation: Vec3::new(
                    10000.0 + (i % 64) as f32 * 3.0,
                    0.0,
                    -((i / 64) as f32) * 3.0,
                ),
                ..Default::default()
            },
        });
    }
    Ok(())
}

impl GraphicsApp {
    /// Manifest reload is explicit and synchronous. Failed CPU preparation or
    /// GPU upload leaves the displayed scene unchanged.
    fn reload_scene(&mut self) -> Result<bool, BoxError> {
        let Some(path) = self.scene_path.clone() else {
            return Ok(false);
        };
        let prepared = match govno_scene::PreparedScene::load(&path, &mut self.model_assets) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.model_assets.remove_unused();
                tracing::warn!(%error, "scene reload rejected; active scene retained");
                return Ok(false);
            }
        };
        if self.expected_offscreen > 0 && prepared.models().is_empty() {
            tracing::warn!("scene reload rejected: culling stress requires geometry");
            self.model_assets.remove_unused();
            return Ok(false);
        }
        // Construct fallible CPU state before allocating GPU resources.
        let mut next = scene_upload::world(&prepared, Vec::new())?;
        let (upload, instances) =
            match scene_upload::SceneUpload::upload(&mut self.renderer, &prepared) {
                Ok(uploaded) => uploaded,
                Err(error) => {
                    drop(prepared);
                    self.model_assets.remove_unused();
                    tracing::warn!(%error, "scene upload failed; active scene retained");
                    return Ok(false);
                }
            };
        next.instances = instances;
        if let Err(error) = add_culling_stress(&mut next, self.expected_offscreen) {
            upload.release(&mut self.renderer)?;
            return Err(error);
        }
        self.world.instances = next.instances;
        self.world.camera = next.camera;
        self.world.light_direction = next.light_direction;
        self.world.light_radiance = next.light_radiance;
        self.world.point_lights = next.point_lights;
        self.simulation.reset_camera(self.world.camera.clone());
        self.simulation.clear();
        install_scene(&mut self.simulation, &prepared, &mut self.world.instances);
        self.simulation.extract(&mut self.world);
        let previous = self.scene_upload.replace(upload);
        self.prepared_scene = Some(prepared);
        if let Some(previous) = previous {
            previous.release(&mut self.renderer)?;
        }
        self.model_assets.remove_unused();

        tracing::info!(path=%path.display(), cached_models=self.model_assets.len(),
            instances=self.world.instances.len(), "authored scene reloaded");
        Ok(true)
    }
}

fn install_scene(
    simulation: &mut govno_ecs::Simulation,
    prepared: &govno_scene::PreparedScene,
    instances: &mut Vec<govno_render::DrawInstance>,
) {
    let mut draws = std::mem::take(instances).into_iter();
    for (model, description) in prepared.models().iter().zip(&prepared.description().models) {
        if let Some(entity) = simulation.spawn_model(draws.by_ref().take(model.data().len()))
            && let Some(body) = &description.physics
        {
            simulation.cuboid(
                entity,
                Vec3::from_array(body.half_extents) * model.transform().scale,
                body.dynamic,
            );
        }
    }
    // Additional app-owned draws, such as the culling stress fixture.
    for instance in draws {
        simulation.spawn(instance);
    }
    if prepared.description().player {
        simulation.enable_player();
    }
}

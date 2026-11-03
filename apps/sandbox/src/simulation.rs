//! Temporary sandbox simulation host; its state will become ECS components/resources.
//! Keeps fixed simulation state separate from interpolated rendering state.
use glam::{Quat, Vec3};
use govno_input::{ActionMap, InputEvent, InputState, actions::*};
use govno_render::Camera;
use govno_runtime::{FixedClock, FixedUpdate};
use std::time::Duration;

pub struct Simulation {
    input: InputState,
    clock: FixedClock,
    initial_camera: Camera,
    camera: Camera,
    previous_position: Vec3,
    previous_seconds: f64,
    seconds: f64,
    alpha: f32,
    reload_scene: bool,
    pub ticks: u64,
    pub discarded: Duration,
}
impl Simulation {
    pub fn new(camera: Camera) -> Self {
        Self {
            input: InputState::new(ActionMap::desktop()),
            clock: FixedClock::default(),
            initial_camera: camera.clone(),
            previous_position: camera.position,
            camera,
            previous_seconds: 0.0,
            seconds: 0.0,
            alpha: 0.0,
            reload_scene: false,
            ticks: 0,
            discarded: Duration::ZERO,
        }
    }

    pub fn input(&mut self, event: InputEvent) {
        self.input.process(event);
        if matches!(event, InputEvent::FocusLost) {
            self.reload_scene = false;
            self.clock.reset();
            self.previous_position = self.camera.position;
            self.previous_seconds = self.seconds;
            self.alpha = 0.0;
        }
    }

    pub fn advance(&mut self, elapsed: Duration) -> FixedUpdate {
        let [x, y] = self.input.take_look();
        self.camera.look(x, y);
        let update = self.clock.advance(elapsed);
        let seconds = self.clock.step().as_secs_f32();
        for _ in 0..update.steps {
            self.previous_position = self.camera.position;
            self.previous_seconds = self.seconds;
            if self.input.just_pressed(RESET) {
                self.camera = self.initial_camera.clone();
                self.previous_position = self.camera.position;
            }
            self.camera.move_local(
                Vec3::new(
                    self.input.axis(LEFT, RIGHT),
                    self.input.axis(DESCEND, RISE),
                    self.input.axis(BACKWARD, FORWARD),
                ),
                seconds,
                if self.input.held(SPRINT) { 10.0 } else { 4.0 },
            );
            self.seconds += self.clock.step().as_secs_f64();
            self.ticks += 1;
            self.reload_scene |= self.input.just_pressed(RELOAD_SCENE);
            self.input.finish_tick();
        }
        self.alpha = update.alpha;
        self.discarded = self.discarded.saturating_add(update.discarded);
        update
    }

    pub fn take_scene_reload(&mut self) -> bool {
        std::mem::take(&mut self.reload_scene)
    }

    pub fn reset_camera(&mut self, camera: Camera) {
        self.initial_camera = camera.clone();
        self.previous_position = camera.position;
        self.camera = camera;
        self.clock.reset();
        self.seconds = 0.0;
        self.previous_seconds = 0.0;
        self.alpha = 0.0;
    }

    pub fn camera(&self) -> Camera {
        let mut camera = self.camera.clone();
        camera.position = self
            .previous_position
            .lerp(self.camera.position, self.alpha);
        camera
    }

    pub fn rotation(&self) -> Quat {
        let seconds =
            self.previous_seconds + (self.seconds - self.previous_seconds) * f64::from(self.alpha);
        Quat::from_rotation_y((seconds * 0.4).rem_euclid(std::f64::consts::TAU) as f32)
    }
}

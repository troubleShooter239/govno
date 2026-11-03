//! ECS simulation and interpolated extraction, independent of Vulkan and windows.
use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use govno_input::{ActionMap, InputEvent, InputState, actions::*};
use govno_render::{Camera, DrawInstance, Material, MeshHandle, RenderWorld, Transform};
use govno_runtime::{FixedClock, FixedUpdate};
use rapier3d::prelude::*;
use std::time::Duration;

#[derive(Component)]
pub struct Spatial {
    pub previous: Transform,
    pub current: Transform,
}
#[derive(Component)]
pub struct Renderable {
    pub mesh: MeshHandle,
    pub material: Material,
}
#[derive(Component)]
struct AdditionalPrimitives(Vec<(MeshHandle, Material)>);
#[derive(Component)]
pub struct Spin(pub f32);
#[derive(Component)]
struct Body(RigidBodyHandle);
#[derive(Resource, Default)]
struct Physics(PhysicsWorld);
#[derive(Resource)]
struct Tick(f32);

fn remember(mut query: Query<&mut Spatial>) {
    for mut s in &mut query {
        s.previous = s.current.clone();
    }
}
fn rotate(tick: Res<Tick>, mut query: Query<(&Spin, &mut Spatial), Without<Body>>) {
    for (spin, mut s) in &mut query {
        s.current.rotation =
            (Quat::from_rotation_y(spin.0 * tick.0) * s.current.rotation).normalize();
    }
}
fn step(mut physics: ResMut<Physics>, tick: Res<Tick>) {
    physics.0.integration_parameters.dt = tick.0;
    physics.0.step();
}
fn synchronize(physics: Res<Physics>, mut query: Query<(&Body, &mut Spatial)>) {
    for (body, mut s) in &mut query {
        if let Some(body) = physics.0.bodies.get(body.0) {
            let p = body.translation();
            let q = body.rotation();
            s.current.translation = Vec3::new(p.x, p.y, p.z);
            s.current.rotation = Quat::from_xyzw(q.x, q.y, q.z, q.w);
        }
    }
}
pub struct Simulation {
    world: World,
    schedule: Schedule,
    extraction: QueryState<(
        &'static Spatial,
        &'static Renderable,
        Option<&'static AdditionalPrimitives>,
    )>,
    input: InputState,
    clock: FixedClock,
    camera: Camera,
    initial_camera: Camera,
    previous_position: Vec3,
    player: Option<RigidBodyHandle>,
    alpha: f32,
    reload_scene: bool,
    pub paused: bool,
    pub ticks: u64,
    pub discarded: Duration,
}
impl Simulation {
    pub fn new(camera: Camera) -> Self {
        let mut world = World::new();
        world.insert_resource(Physics::default());
        world.insert_resource(Tick(1.0 / 60.0));
        let extraction = world.query();
        let mut schedule = Schedule::default();
        schedule.add_systems((remember, rotate, step, synchronize).chain());
        Self {
            world,
            schedule,
            extraction,
            input: InputState::new(ActionMap::desktop()),
            clock: FixedClock::default(),
            previous_position: camera.position,
            initial_camera: camera.clone(),
            camera,
            player: None,
            alpha: 0.0,
            reload_scene: false,
            paused: false,
            ticks: 0,
            discarded: Duration::ZERO,
        }
    }
    pub fn spawn(&mut self, instance: DrawInstance) -> Entity {
        self.world
            .spawn((
                Spatial {
                    previous: instance.transform.clone(),
                    current: instance.transform,
                },
                Renderable {
                    mesh: instance.mesh,
                    material: instance.material,
                },
            ))
            .id()
    }
    pub fn spin(&mut self, entity: Entity, speed: f32) {
        self.world.entity_mut(entity).insert(Spin(speed));
    }
    /// All primitives of one model share a single transform and rigid body.
    pub fn spawn_model(
        &mut self,
        mut instances: impl Iterator<Item = DrawInstance>,
    ) -> Option<Entity> {
        let entity = self.spawn(instances.next()?);
        self.world.entity_mut(entity).insert(AdditionalPrimitives(
            instances.map(|i| (i.mesh, i.material)).collect(),
        ));
        Some(entity)
    }
    pub fn sphere(&mut self, entity: Entity, radius: f32) {
        self.remove_body(entity);
        let p = self
            .world
            .get::<Spatial>(entity)
            .expect("spatial entity")
            .current
            .translation;
        let (h, _) = self.world.resource_mut::<Physics>().0.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(p.x, p.y, p.z)),
            ColliderBuilder::ball(radius),
        );
        self.world.entity_mut(entity).insert(Body(h));
    }
    /// Shapes are explicitly supplied by the game, never inferred from render bounds.
    pub fn cuboid(&mut self, entity: Entity, half: Vec3, dynamic: bool) {
        self.remove_body(entity);
        let t = self
            .world
            .get::<Spatial>(entity)
            .expect("spatial entity")
            .current
            .clone();
        let angle = t.rotation.to_scaled_axis();
        let builder = if dynamic {
            RigidBodyBuilder::dynamic()
        } else {
            RigidBodyBuilder::fixed()
        };
        let (handle, _) = self.world.resource_mut::<Physics>().0.insert(
            builder
                .translation(Vector::new(
                    t.translation.x,
                    t.translation.y,
                    t.translation.z,
                ))
                .rotation(Vector::new(angle.x, angle.y, angle.z))
                .ccd_enabled(dynamic),
            ColliderBuilder::cuboid(half.x, half.y, half.z).friction(0.7),
        );
        self.world
            .entity_mut(entity)
            .remove::<Spin>()
            .insert(Body(handle));
    }
    fn remove_body(&mut self, entity: Entity) {
        if let Some(body) = self.world.get::<Body>(entity) {
            let handle = body.0;
            self.world.resource_mut::<Physics>().0.remove_body(handle);
            self.world.entity_mut(entity).remove::<Body>();
        }
    }
    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.remove_body(entity);
        self.world.despawn(entity)
    }
    pub fn clear(&mut self) {
        // Bevy 0.20 also stores resources as entities; remove scene entities only.
        let entities: Vec<_> = self
            .world
            .query_filtered::<Entity, With<Spatial>>()
            .iter(&self.world)
            .collect();
        for entity in entities {
            self.world.despawn(entity);
        }
        self.world.insert_resource(Physics::default());
        self.player = None;
        self.clock.reset();
        self.alpha = 0.0;
    }
    pub fn enable_player(&mut self) {
        if self.player.is_some() {
            return;
        }
        let p = self.camera.position - Vec3::Y * 0.6;
        let (h, _) = self.world.resource_mut::<Physics>().0.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(p.x, p.y, p.z))
                .lock_rotations()
                .ccd_enabled(true),
            ColliderBuilder::capsule_y(0.5, 0.3).friction(0.0),
        );
        self.player = Some(h);
    }
    pub fn input(&mut self, event: InputEvent) {
        self.input.process(event);
        if matches!(event, InputEvent::FocusLost) {
            self.clock.reset();
            self.previous_position = self.camera.position;
            self.alpha = 0.0;
            self.reload_scene = false;
        }
    }
    fn control_player(&mut self) {
        let Some(h) = self.player else {
            return;
        };
        let mut physics = self.world.resource_mut::<Physics>();
        let p = physics.0.bodies[h].translation();
        let grounded = physics
            .0
            .cast_ray_and_get_normal(
                &Ray::new(p, -Vector::Y),
                0.87,
                true,
                QueryFilter::default().exclude_rigid_body(h),
            )
            .is_some_and(|(_, hit)| hit.normal.y > 0.6);
        let right = Vec3::new(self.camera.yaw.cos(), 0.0, self.camera.yaw.sin());
        let forward = Vec3::new(self.camera.yaw.sin(), 0.0, -self.camera.yaw.cos());
        let direction = (right * self.input.axis(LEFT, RIGHT)
            + forward * self.input.axis(BACKWARD, FORWARD))
        .normalize_or_zero();
        let speed = if self.input.held(SPRINT) { 7.0 } else { 4.0 };
        let body = &mut physics.0.bodies[h];
        let mut v = body.linvel();
        v.x = direction.x * speed;
        v.z = direction.z * speed;
        if grounded && self.input.just_pressed(JUMP) {
            v.y = 5.0;
        }
        if self.input.just_pressed(RESET) || p.y < -30.0 {
            let p = self.initial_camera.position - Vec3::Y * 0.6;
            body.set_translation(Vector::new(p.x, p.y, p.z), true);
            v = Vector::ZERO;
            self.camera = self.initial_camera.clone();
            self.previous_position = self.camera.position;
        }
        body.set_linvel(v, true);
        if self.input.just_pressed(INTERACT) {
            let p = self.camera.position;
            let d = self.camera.forward();
            if let Some((c, _)) = physics.0.cast_ray(
                &Ray::new(Vector::new(p.x, p.y, p.z), Vector::new(d.x, d.y, d.z)),
                4.0,
                true,
                QueryFilter::default().exclude_rigid_body(h),
            ) && let Some(target) = physics.0.colliders[c].parent()
            {
                physics.0.bodies[target].apply_impulse(Vector::new(d.x, d.y, d.z) * 12.0, true);
            }
        }
    }
    pub fn advance(&mut self, elapsed: Duration) -> FixedUpdate {
        let [x, y] = self.input.take_look();
        self.camera.look(x, y);
        if self.paused {
            self.reload_scene |= self.input.just_pressed(RELOAD_SCENE);
            self.clock.reset();
            self.input.finish_tick();
            return FixedUpdate {
                steps: 0,
                alpha: self.alpha,
                discarded: Duration::ZERO,
            };
        }
        let update = self.clock.advance(elapsed);
        for _ in 0..update.steps {
            self.reload_scene |= self.input.just_pressed(RELOAD_SCENE);
            self.previous_position = self.camera.position;
            if self.player.is_some() {
                self.control_player();
            } else {
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
                    self.clock.step().as_secs_f32(),
                    if self.input.held(SPRINT) { 10.0 } else { 4.0 },
                );
            }
            self.world.resource_mut::<Tick>().0 = self.clock.step().as_secs_f32();
            self.schedule.run(&mut self.world);
            if let Some(h) = self.player {
                let p = self.world.resource::<Physics>().0.bodies[h].translation();
                self.camera.position = Vec3::new(p.x, p.y + 0.6, p.z);
            }
            self.ticks += 1;
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
        self.alpha = 0.0;
    }
    pub fn extract(&mut self, render: &mut RenderWorld) {
        render.camera = self.camera.clone();
        render.camera.position = self
            .previous_position
            .lerp(self.camera.position, self.alpha);
        render.instances.clear();
        for (s, r, additional) in self.extraction.iter(&self.world) {
            let transform = Transform {
                translation: s
                    .previous
                    .translation
                    .lerp(s.current.translation, self.alpha),
                rotation: s.previous.rotation.slerp(s.current.rotation, self.alpha),
                scale: s.previous.scale.lerp(s.current.scale, self.alpha),
            };
            render.instances.push(DrawInstance {
                mesh: r.mesh,
                material: r.material,
                transform: transform.clone(),
            });
            if let Some(additional) = additional {
                render
                    .instances
                    .extend(additional.0.iter().map(|&(mesh, material)| DrawInstance {
                        mesh,
                        material,
                        transform: transform.clone(),
                    }));
            }
        }
    }
    pub fn body_count(&self) -> usize {
        self.world.resource::<Physics>().0.bodies.len()
    }
    pub fn entity_count(&mut self) -> u32 {
        self.extraction.iter(&self.world).count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn instance(position: Vec3) -> DrawInstance {
        DrawInstance {
            mesh: MeshHandle::from_parts(1, 0),
            transform: Transform {
                translation: position,
                ..Default::default()
            },
            material: Material {
                emissive_texture: None,
                occlusion_texture: None,
                emissive: Vec3::ZERO,
                occlusion_strength: 1.0,
                normal_texture: None,
                metallic_roughness_texture: None,
                normal_scale: 1.0,
                base_color_texture: None,
                base_color: Vec3::ONE,
                metallic: 0.0,
                roughness: 0.5,
            },
        }
    }
    fn tick(s: &mut Simulation, n: usize) {
        for _ in 0..n {
            s.advance(Duration::from_nanos(16_666_667));
        }
    }
    fn floor(s: &mut Simulation) {
        let e = s.spawn(instance(Vec3::new(0.0, -0.5, 0.0)));
        s.cuboid(e, Vec3::new(10.0, 0.5, 10.0), false);
    }
    #[test]
    fn rigid_body_settles_and_despawn_removes_collider() {
        let mut s = Simulation::new(Camera::default());
        floor(&mut s);
        let e = s.spawn(instance(Vec3::new(0.0, 5.0, 0.0)));
        s.cuboid(e, Vec3::splat(0.5), true);
        tick(&mut s, 180);
        let y = s.world.get::<Spatial>(e).unwrap().current.translation.y;
        assert!((y - 0.5).abs() < 0.03, "body must rest on floor: {y}");
        assert!(s.despawn(e));
        assert_eq!(s.body_count(), 1);
        assert_eq!(s.world.resource::<Physics>().0.colliders.len(), 1);
        s.clear();
        assert_eq!(s.entity_count(), 0);
        assert_eq!(s.body_count(), 0);
        floor(&mut s);
        tick(&mut s, 1);
        assert_eq!(s.body_count(), 1);
    }
    #[test]
    fn player_lands_jumps_once_and_collides_with_wall() {
        let mut s = Simulation::new(Camera {
            position: Vec3::new(0.0, 3.0, 3.0),
            ..Default::default()
        });
        floor(&mut s);
        let wall = s.spawn(instance(Vec3::new(0.0, 1.0, 0.0)));
        s.cuboid(wall, Vec3::new(2.0, 1.0, 0.2), false);
        s.enable_player();
        tick(&mut s, 120);
        let landed = s.camera.position.y;
        assert!((landed - 1.4).abs() < 0.04, "landed: {landed}");
        s.input(InputEvent::Key {
            key: govno_input::InputKey::Space,
            pressed: true,
        });
        tick(&mut s, 10);
        assert!(s.camera.position.y > landed + 0.3);
        tick(&mut s, 120);
        assert!(
            (s.camera.position.y - landed).abs() < 0.04,
            "held jump must not repeat"
        );
        s.input(InputEvent::Key {
            key: govno_input::InputKey::W,
            pressed: true,
        });
        tick(&mut s, 120);
        assert!(
            s.camera.position.z >= 0.45,
            "player tunneled through wall: {}",
            s.camera.position.z
        );
    }
    #[test]
    fn pause_and_reload_preserve_resources_and_remove_stale_bodies() {
        let mut s = Simulation::new(Camera::default());
        floor(&mut s);
        s.enable_player();
        tick(&mut s, 10);
        let before = s.camera.position;
        s.paused = true;
        tick(&mut s, 60);
        assert_eq!(before, s.camera.position);
        assert_eq!(s.ticks, 10);
        s.clear();
        floor(&mut s);
        s.enable_player();
        s.paused = false;
        tick(&mut s, 1);
        assert_eq!(s.body_count(), 2);
        assert_eq!(s.world.resource::<Physics>().0.colliders.len(), 2);
    }
}

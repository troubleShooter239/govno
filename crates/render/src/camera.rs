use crate::RenderExtent;
use glam::{Mat4, Vec3};
use govno_core::BoxError;

/// Right-handed, Y-up, yaw=0 looks down -Z. Angles in radians.
#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub vertical_fov: f32,
    pub near: f32,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 7.0),
            yaw: 0.0,
            pitch: -0.12,
            vertical_fov: 60.0_f32.to_radians(),
            near: 0.05,
        }
    }
}
impl Camera {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
    }
    pub fn look(&mut self, x: f32, y: f32) {
        if !x.is_finite() || !y.is_finite() {
            return;
        }
        self.yaw = (self.yaw + x * 0.0025).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch - y * 0.0025).clamp(-1.553, 1.553);
    }
    /// Local x=right, y=world-up, z=forward. Diagonal movement is normalized.
    pub fn move_local(&mut self, direction: Vec3, seconds: f32, speed: f32) {
        if !direction.is_finite() || !seconds.is_finite() || !speed.is_finite() {
            return;
        }
        let right = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin());
        let movement = right * direction.x + Vec3::Y * direction.y + self.forward() * direction.z;
        self.position += movement.normalize_or_zero() * seconds.clamp(0.0, 0.1) * speed.max(0.0);
    }
    pub fn view_projection(&self, extent: RenderExtent) -> Result<Mat4, BoxError> {
        if extent.is_zero()
            || !self.position.is_finite()
            || !self.yaw.is_finite()
            || !self.pitch.is_finite()
            || self.pitch.abs() >= std::f32::consts::FRAC_PI_2
            || !self.vertical_fov.is_finite()
            || !(0.0..std::f32::consts::PI).contains(&self.vertical_fov)
            || self.vertical_fov == 0.0
            || !self.near.is_finite()
            || self.near <= 0.0
        {
            return Err("invalid camera or render extent".into());
        }
        let projection = glam::camera::rh::proj::vulkan::perspective_infinite_reverse(
            self.vertical_fov,
            extent.width as f32 / extent.height as f32,
            self.near,
        );
        let view = glam::camera::rh::view::look_to_mat4(self.position, self.forward(), Vec3::Y);
        let matrix = projection * view;
        if !matrix.is_finite() {
            return Err("camera view-projection exceeds finite f32 range".into());
        }
        Ok(matrix)
    }
}

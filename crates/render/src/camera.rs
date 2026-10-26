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
        Ok(projection * view)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn camera() -> Camera {
        Camera {
            position: Vec3::ZERO,
            pitch: 0.0,
            ..Default::default()
        }
    }
    #[test]
    fn reversed_z_and_vulkan_y_are_consistent() {
        let camera = camera();
        let matrix = camera
            .view_projection(RenderExtent {
                width: 100,
                height: 100,
            })
            .unwrap();
        let near = matrix.project_point3(Vec3::new(0.0, 0.0, -camera.near));
        let far = matrix.project_point3(Vec3::new(0.0, 0.0, -10000.0));
        assert!((near.z - 1.0).abs() < 1e-5);
        assert!(far.z > 0.0 && far.z < 0.0001);
        assert!(matrix.project_point3(Vec3::new(0.0, 1.0, -2.0)).y < 0.0);
    }
    #[test]
    fn aspect_ratio_only_changes_horizontal_scale() {
        let camera = camera();
        let square = camera
            .view_projection(RenderExtent {
                width: 100,
                height: 100,
            })
            .unwrap();
        let wide = camera
            .view_projection(RenderExtent {
                width: 200,
                height: 100,
            })
            .unwrap();
        assert!((wide.x_axis.x * 2.0 - square.x_axis.x).abs() < 1e-5);
        assert_eq!(wide.y_axis.y, square.y_axis.y);
    }
    #[test]
    fn movement_and_look_are_bounded() {
        let mut straight = camera();
        let mut diagonal = camera();
        straight.move_local(Vec3::Z, 0.1, 4.0);
        diagonal.move_local(Vec3::new(1.0, 0.0, 1.0), 0.1, 4.0);
        assert!((straight.position.length() - diagonal.position.length()).abs() < 1e-6);
        straight.look(0.0, 1e6);
        assert!(straight.pitch > -std::f32::consts::FRAC_PI_2);
        assert!(
            straight
                .view_projection(RenderExtent {
                    width: 100,
                    height: 100
                })
                .is_ok()
        );
    }
    #[test]
    fn invalid_projection_is_rejected() {
        let mut camera = camera();
        assert!(camera.view_projection(RenderExtent::default()).is_err());
        camera.near = 0.0;
        assert!(
            camera
                .view_projection(RenderExtent {
                    width: 100,
                    height: 100
                })
                .is_err()
        );
    }
}

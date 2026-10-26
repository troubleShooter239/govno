//! Explicit single-map directional shadow volume.
use glam::{Mat4, Vec3};
use govno_core::BoxError;
#[derive(Debug, Clone, Copy)]
pub struct DirectionalShadow {
    pub center: Vec3,
    /// Half width/height of the orthographic volume, in scene units.
    pub half_extent: f32,
    /// Receiver depth bias in normalized shadow depth.
    pub bias: f32,
}
impl Default for DirectionalShadow {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            half_extent: 12.0,
            bias: 0.0005,
        }
    }
}
impl DirectionalShadow {
    pub fn view_projection(self, direction_to_light: Vec3) -> Result<Mat4, BoxError> {
        if !self.center.is_finite()
            || !self.half_extent.is_finite()
            || !(0.1..=10000.0).contains(&self.half_extent)
            || !self.bias.is_finite()
            || !(0.0..=0.05).contains(&self.bias)
            || !direction_to_light.is_finite()
            || !direction_to_light.length_squared().is_finite()
            || direction_to_light.length_squared() < 1e-8
        {
            return Err("invalid directional shadow volume, bias or light direction".into());
        }
        let direction = direction_to_light.normalize();
        let up = if direction.dot(Vec3::Y).abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let view = glam::camera::rh::view::look_at_mat4(
            self.center + direction * (2.0 * self.half_extent),
            self.center,
            up,
        );
        let e = self.half_extent;
        // Conventional finite 0..1 depth for shadows, independent of main reversed-Z.
        let projection = glam::camera::rh::proj::vulkan::orthographic(-e, e, -e, e, 0.01, 4.0 * e);

        let matrix = projection * view;
        if !matrix.is_finite() {
            return Err("shadow matrix is non-finite".into());
        }
        Ok(matrix)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shadow_projection_uses_finite_depth_and_handles_vertical_light() {
        let s = DirectionalShadow::default();
        for dir in [Vec3::Y, -Vec3::Y, Vec3::new(-0.5, 1.0, 0.7).normalize()] {
            let m = s.view_projection(dir).unwrap();
            let center = m.project_point3(s.center);
            assert!(center.x.abs() < 1e-5 && center.y.abs() < 1e-5);
            assert!((0.0..1.0).contains(&center.z));
            assert!(m.project_point3(s.center + dir).z < center.z);
            assert!(m.project_point3(s.center - dir).z > center.z);
        }
        let m = s.view_projection(Vec3::Z).unwrap();
        assert!(m.project_point3(Vec3::Y).y < 0.0);
        assert!(s.view_projection(Vec3::ZERO).is_err());
        for extent in [0.0, f32::NAN, f32::INFINITY, 10001.0] {
            assert!(
                DirectionalShadow {
                    half_extent: extent,
                    ..s
                }
                .view_projection(Vec3::Y)
                .is_err()
            );
        }
        assert!(
            DirectionalShadow { bias: -0.1, ..s }
                .view_projection(Vec3::Y)
                .is_err()
        );
    }
}

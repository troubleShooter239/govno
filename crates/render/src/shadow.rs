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
    pub distance: f32,
    pub split_lambda: f32,
    pub cascade_blend: f32,
    pub debug_cascades: bool,
}
impl Default for DirectionalShadow {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            half_extent: 12.0,
            bias: 0.0005,
            distance: 80.0,
            split_lambda: 0.65,
            cascade_blend: 0.1,
            debug_cascades: false,
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

pub const CASCADE_COUNT: usize = 4;
pub struct ShadowCascades {
    pub matrices: [Mat4; 4],
    pub splits: [f32; 4],
}
impl DirectionalShadow {
    pub fn cascades(
        self,
        camera: &crate::Camera,
        extent: crate::RenderExtent,
        light: Vec3,
    ) -> Result<ShadowCascades, BoxError> {
        camera.view_projection(extent)?;
        self.view_projection(light)?;
        if ![self.distance, self.split_lambda, self.cascade_blend]
            .iter()
            .all(|v| v.is_finite())
            || self.distance <= camera.near
            || self.distance > 10000.0
            || !(0.0..=1.0).contains(&self.split_lambda)
            || !(0.0..=0.3).contains(&self.cascade_blend)
        {
            return Err("invalid shadow cascade distance/splits/blend".into());
        }
        let near = camera.near;
        let far = self.distance;
        let splits = std::array::from_fn(|i| {
            let fraction = (i + 1) as f32 / 4.0;
            self.split_lambda * near * (far / near).powf(fraction)
                + (1.0 - self.split_lambda) * (near + (far - near) * fraction)
        });
        let forward = camera.forward();
        let right = Vec3::new(camera.yaw.cos(), 0.0, camera.yaw.sin());
        let up = right.cross(forward);
        let direction = light.normalize();
        let light_up = if direction.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let mut start = near;
        let matrices = std::array::from_fn(|i| {
            let end = splits[i];
            // Include the preceding cascade's blend interval in this cascade's volume.
            let begin = if i > 0 {
                start - (start - if i > 1 { splits[i - 2] } else { near }) * self.cascade_blend
            } else {
                near
            };
            let mut corners = [Vec3::ZERO; 8];
            for (plane, depth) in [begin, end].into_iter().enumerate() {
                let half = depth * (camera.vertical_fov * 0.5).tan();
                for (j, (x, y)) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
                    .into_iter()
                    .enumerate()
                {
                    corners[plane * 4 + j] = camera.position
                        + forward * depth
                        + right * x * half * extent.width as f32 / extent.height as f32
                        + up * y * half;
                }
            }
            let center = corners.iter().copied().sum::<Vec3>() / 8.0;
            let radius = (corners
                .iter()
                .map(|p| p.distance(center))
                .fold(0.0_f32, f32::max)
                * 16.0)
                .ceil()
                / 16.0;
            let padding = self.half_extent;
            let view = glam::camera::rh::view::look_at_mat4(
                center + direction * (2.0 * radius + padding),
                center,
                light_up,
            );
            let mut projection = glam::camera::rh::proj::vulkan::orthographic(
                -radius,
                radius,
                -radius,
                radius,
                0.01,
                4.0 * radius + 2.0 * padding,
            );
            let origin = (projection * view).project_point3(Vec3::ZERO) * 1024.0;
            projection.w_axis.x += (origin.x.round() - origin.x) / 1024.0;
            projection.w_axis.y += (origin.y.round() - origin.y) / 1024.0;
            start = end;
            projection * view
        });
        if matrices.iter().any(|m| !m.is_finite()) {
            return Err("cascade matrix is non-finite".into());
        }
        Ok(ShadowCascades { matrices, splits })
    }
}
#[cfg(test)]
mod cascade_tests {
    use super::*;
    #[test]
    fn practical_splits_cover_finite_view_and_track_camera() {
        let s = DirectionalShadow::default();
        let camera = crate::Camera::default();
        let extent = crate::RenderExtent {
            width: 1280,
            height: 720,
        };
        let a = s.cascades(&camera, extent, Vec3::Y).unwrap();
        assert!(a.splits.windows(2).all(|w| w[1] > w[0]));
        assert!((a.splits[3] - s.distance).abs() < 0.001);
        let mut moved = camera.clone();
        moved.position.x += 8.0;
        let b = s.cascades(&moved, extent, Vec3::Y).unwrap();
        assert_ne!(a.matrices[0], b.matrices[0]);
        for (i, m) in a.matrices.iter().enumerate() {
            let begin = if i == 0 { camera.near } else { a.splits[i - 1] };
            let p =
                m.project_point3(camera.position + camera.forward() * (begin + a.splits[i]) * 0.5);
            assert!(p.x.abs() < 1.0 && p.y.abs() < 1.0 && (0.0..1.0).contains(&p.z));
        }
    }
}

#[cfg(test)]
mod containment_tests {
    use super::*;
    #[test]
    fn cascade_matrices_contain_all_slice_corners() {
        let camera = crate::Camera::default();
        let extent = crate::RenderExtent {
            width: 1280,
            height: 720,
        };
        let s = DirectionalShadow::default();
        let c = s
            .cascades(&camera, extent, Vec3::new(-0.5, 1.0, 0.7))
            .unwrap();
        let forward = camera.forward();
        let right = Vec3::new(camera.yaw.cos(), 0.0, camera.yaw.sin());
        let up = right.cross(forward);
        for i in 0..4 {
            let begin = if i == 0 { camera.near } else { c.splits[i - 1] };
            for depth in [begin, c.splits[i]] {
                let half = depth * (camera.vertical_fov * 0.5).tan();
                for x in [-1.0, 1.0] {
                    for y in [-1.0, 1.0] {
                        let corner = camera.position
                            + forward * depth
                            + right * x * half * extent.width as f32 / extent.height as f32
                            + up * y * half;
                        let clip = c.matrices[i].project_point3(corner);
                        assert!(
                            clip.x.abs() <= 1.001
                                && clip.y.abs() <= 1.001
                                && (0.0..1.0).contains(&clip.z)
                        );
                    }
                }
            }
        }
    }
}

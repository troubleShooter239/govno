//! Direct-light data shared by renderer backends.
use glam::Vec3;
use govno_core::BoxError;

/// Current basic Forward renderer contract; clustered lighting is a later stage.
pub const MAX_POINT_LIGHTS: usize = 16;
#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    pub position: Vec3,
    /// Nonnegative linear RGB radiant intensity in scene units (not lumens).
    pub intensity: Vec3,
    /// Finite influence radius, at least 0.01 scene units.
    pub range: f32,
}
impl PointLight {
    pub fn validate(self) -> Result<(), BoxError> {
        if !self.position.is_finite()
            || !self.intensity.is_finite()
            || self.intensity.min_element() < 0.0
            || !self.range.is_finite()
            || self.range < 0.01
        {
            return Err(
                "point light needs finite position, nonnegative intensity and range >= 0.01".into(),
            );
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn world(lights: Vec<PointLight>) -> crate::RenderWorld {
        crate::RenderWorld {
            post_process: Default::default(),
            directional_shadow: None,
            diffuse_environment: None,
            specular_environment: false,
            local_probes: vec![],
            camera: Default::default(),
            light_direction: Vec3::Y,
            light_radiance: Vec3::ONE,
            point_lights: lights,
            instances: vec![],
        }
    }
    #[test]
    fn world_checks_light_count_and_every_light() {
        let light = PointLight {
            position: Vec3::ZERO,
            intensity: Vec3::ONE,
            range: 4.0,
        };
        let extent = crate::RenderExtent {
            width: 1,
            height: 1,
        };
        for count in [0, 1, MAX_POINT_LIGHTS] {
            assert!(world(vec![light; count]).validate(extent).is_ok());
        }
        assert!(
            world(vec![light; MAX_POINT_LIGHTS + 1])
                .validate(extent)
                .is_err()
        );
        for bad in [
            PointLight {
                position: Vec3::splat(f32::NAN),
                ..light
            },
            PointLight {
                intensity: Vec3::splat(-0.1),
                ..light
            },
            PointLight {
                intensity: Vec3::splat(f32::INFINITY),
                ..light
            },
            PointLight {
                range: 0.0,
                ..light
            },
            PointLight {
                range: f32::NAN,
                ..light
            },
            PointLight {
                range: f32::INFINITY,
                ..light
            },
        ] {
            assert!(world(vec![light, bad]).validate(extent).is_err());
        }
        assert!(
            world(vec![PointLight {
                intensity: Vec3::ZERO,
                ..light
            }])
            .validate(extent)
            .is_ok()
        );
    }
}

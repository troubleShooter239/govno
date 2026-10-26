//! CPU-side post-processing controls, independent of the rendering backend.
use govno_core::BoxError;

#[derive(Debug, Clone, Copy, Default)]
pub struct PostProcessSettings {
    /// Exposure compensation in stops: +1 doubles linear scene brightness.
    /// Accepted range is [-16, 16]; zero preserves the original scene exposure.
    pub exposure_ev: f32,
}
impl PostProcessSettings {
    pub fn validate(self) -> Result<(), BoxError> {
        if !self.exposure_ev.is_finite() || !(-16.0..=16.0).contains(&self.exposure_ev) {
            return Err("exposure must be finite and within [-16, 16] EV".into());
        }
        Ok(())
    }
    /// Linear multiplier to apply before tone mapping. Validate settings first.
    pub fn exposure_multiplier(self) -> f32 {
        self.exposure_ev.exp2()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exposure_stops_preserve_linear_ratios() {
        assert_eq!(PostProcessSettings::default().exposure_multiplier(), 1.0);
        for (ev, expected) in [
            (-16.0, 1.0 / 65536.0),
            (-1.0, 0.5),
            (1.0, 2.0),
            (16.0, 65536.0),
        ] {
            let settings = PostProcessSettings { exposure_ev: ev };
            settings.validate().unwrap();
            assert_eq!(settings.exposure_multiplier(), expected);
        }
    }
    #[test]
    fn invalid_exposure_is_rejected_by_render_world() {
        for ev in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -16.01, 16.01] {
            let world = crate::RenderWorld {
                post_process: PostProcessSettings { exposure_ev: ev },
                directional_shadow: None,
                camera: Default::default(),
                light_direction: glam::Vec3::Y,
                light_radiance: glam::Vec3::ONE,
                point_lights: vec![],
                instances: vec![],
            };
            assert!(
                world
                    .validate(crate::RenderExtent {
                        width: 1,
                        height: 1
                    })
                    .is_err()
            );
        }
    }
}

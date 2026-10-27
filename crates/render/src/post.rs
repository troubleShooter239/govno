//! Backend-independent post controls.
use govno_core::BoxError;
#[derive(Debug, Clone, Copy)]
pub struct BloomSettings {
    pub intensity: f32,
    pub threshold: f32,
    pub knee: f32,
}
impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            intensity: 0.08,
            threshold: 1.0,
            knee: 0.5,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct AutoExposure {
    pub key: f32,
    pub min_ev: f32,
    pub max_ev: f32,
    pub speed: f32,
}
impl Default for AutoExposure {
    fn default() -> Self {
        Self {
            key: 0.18,
            min_ev: -8.0,
            max_ev: 8.0,
            speed: 1.5,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct PostProcessSettings {
    pub exposure_ev: f32,
    pub bloom: BloomSettings,
    pub auto_exposure: Option<AutoExposure>,
    pub delta_seconds: f32,
}
impl Default for PostProcessSettings {
    fn default() -> Self {
        Self {
            exposure_ev: 0.0,
            bloom: Default::default(),
            auto_exposure: Some(Default::default()),
            delta_seconds: 1.0 / 60.0,
        }
    }
}
impl PostProcessSettings {
    pub fn validate(self) -> Result<(), BoxError> {
        let b = self.bloom;
        if !self.exposure_ev.is_finite()
            || !(-16.0..=16.0).contains(&self.exposure_ev)
            || !self.delta_seconds.is_finite()
            || !(0.0..=1.0).contains(&self.delta_seconds)
            || ![b.intensity, b.threshold, b.knee]
                .iter()
                .all(|v| v.is_finite())
            || !(0.0..=10.0).contains(&b.intensity)
            || !(0.0..=65504.0).contains(&b.threshold)
            || !(0.0..=65504.0).contains(&b.knee)
        {
            return Err("invalid exposure, bloom or delta time".into());
        }
        if let Some(a) = self.auto_exposure
            && (![a.key, a.min_ev, a.max_ev, a.speed]
                .iter()
                .all(|v| v.is_finite())
                || a.key <= 0.0
                || a.min_ev < -16.0
                || a.max_ev > 16.0
                || a.min_ev > a.max_ev
                || a.speed <= 0.0)
        {
            return Err("invalid auto exposure settings".into());
        }
        Ok(())
    }
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
            let settings = PostProcessSettings {
                exposure_ev: ev,
                ..Default::default()
            };
            settings.validate().unwrap();
            assert_eq!(settings.exposure_multiplier(), expected);
        }
    }
    #[test]
    fn invalid_exposure_is_rejected_by_render_world() {
        for ev in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -16.01, 16.01] {
            let world = crate::RenderWorld {
                frustum_culling: true,
                post_process: PostProcessSettings {
                    exposure_ev: ev,
                    ..Default::default()
                },
                directional_shadow: None,
                diffuse_environment: None,
                specular_environment: false,
                local_probes: vec![],
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

#[cfg(test)]
mod controls_tests {
    use super::*;
    #[test]
    fn post_controls_reject_invalid_adaptation_and_bloom() {
        let mut p = PostProcessSettings::default();
        p.bloom.intensity = -1.0;
        assert!(p.validate().is_err());
        p = Default::default();
        p.auto_exposure = Some(AutoExposure {
            min_ev: 4.0,
            max_ev: 1.0,
            ..Default::default()
        });
        assert!(p.validate().is_err());
        p = Default::default();
        p.delta_seconds = f32::NAN;
        assert!(p.validate().is_err());
        p = Default::default();
        p.bloom.knee = f32::INFINITY;
        assert!(p.validate().is_err());
    }
}

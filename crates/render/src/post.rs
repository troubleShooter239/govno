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

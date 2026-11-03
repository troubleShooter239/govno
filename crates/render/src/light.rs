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

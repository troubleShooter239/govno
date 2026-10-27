use glam::Vec3;
use govno_core::BoxError;
pub const MAX_LOCAL_PROBES: usize = 2;
/// Baked HDR light probe. Slots 1 and 2 complement global environment slot 0.
#[derive(Debug, Clone, Copy)]
pub struct LocalProbe {
    pub slot: u32,
    pub position: Vec3,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub fade: f32,
    pub strength: f32,
}
impl LocalProbe {
    pub fn validate(self) -> Result<(), BoxError> {
        let size = self.bounds_max - self.bounds_min;
        if !(1..=2).contains(&self.slot)
            || ![self.position, self.bounds_min, self.bounds_max, size]
                .iter()
                .all(|v| v.is_finite())
            || size.min_element() < 0.01
            || self.position.cmple(self.bounds_min).any()
            || self.position.cmpge(self.bounds_max).any()
            || !self.fade.is_finite()
            || self.fade <= 0.0
            || !self.strength.is_finite()
            || !(0.0..=1.0).contains(&self.strength)
        {
            return Err("invalid local HDR probe slot/box/fade/strength".into());
        }
        Ok(())
    }
    pub fn influence(self, point: Vec3) -> f32 {
        let d = (point - self.bounds_min)
            .min(self.bounds_max - point)
            .min_element();
        let t = (d / self.fade).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t) * self.strength
    }
    pub fn project_reflection(self, point: Vec3, ray: Vec3) -> Vec3 {
        if self.influence(point) == 0.0 {
            return ray;
        }
        let mut distance = f32::INFINITY;
        for i in 0..3 {
            if ray[i].abs() > 1e-6 {
                let boundary = if ray[i] > 0.0 {
                    self.bounds_max[i]
                } else {
                    self.bounds_min[i]
                };
                distance = distance.min((boundary - point[i]) / ray[i]);
            }
        }
        if !distance.is_finite() {
            return ray;
        }
        (point + ray * distance - self.position).normalize_or_zero()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probe_box_fades_and_corrects_parallax() {
        let p = LocalProbe {
            slot: 1,
            position: Vec3::ZERO,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            fade: 1.0,
            strength: 1.0,
        };
        p.validate().unwrap();
        assert_eq!(p.influence(Vec3::ZERO), 1.0);
        assert_eq!(p.influence(Vec3::X * 3.0), 0.0);
        assert_eq!(p.influence(Vec3::X * 1.5), 0.5);
        assert!(
            p.project_reflection(Vec3::X, Vec3::Z)
                .abs_diff_eq(Vec3::new(1.0, 0.0, 2.0).normalize(), 1e-6)
        );
        assert!(LocalProbe { slot: 0, ..p }.validate().is_err());
    }
}

/// Normalized global/first/second local weights in the current probe-list order.
pub fn probe_weights(probes: &[LocalProbe], point: Vec3) -> [f32; 3] {
    let a = probes.first().map_or(0.0, |p| p.influence(point));
    let b = probes.get(1).map_or(0.0, |p| p.influence(point));
    let global = (1.0 - a - b).max(0.0);
    let sum = (global + a + b).max(0.00001);
    [global / sum, a / sum, b / sum]
}
#[cfg(test)]
mod blending_tests {
    use super::*;
    #[test]
    fn overlapping_probes_normalize_and_outside_falls_back() {
        let p = LocalProbe {
            slot: 1,
            position: Vec3::ZERO,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            fade: 1.0,
            strength: 1.0,
        };
        assert_eq!(
            probe_weights(&[p, LocalProbe { slot: 2, ..p }], Vec3::ZERO),
            [0.0, 0.5, 0.5]
        );
        assert_eq!(probe_weights(&[p], Vec3::X * 3.0), [1.0, 0.0, 0.0]);
    }
}

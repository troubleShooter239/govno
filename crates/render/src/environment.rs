//! Low-frequency diffuse environment lighting, independent of image/GPU backends.
use glam::Vec3;
use govno_core::BoxError;
const Y00: f32 = 0.2820948;

/// Nine real SH coefficients of irradiance/pi (linear RGB, world coordinates).
/// The Lambert cosine convolution is already applied; these are not radiance SH.
#[derive(Debug, Clone)]
pub struct DiffuseEnvironment {
    coefficients: [Vec3; 9],
}
impl DiffuseEnvironment {
    pub fn constant(radiance: Vec3) -> Result<Self, BoxError> {
        if !radiance.is_finite() || radiance.min_element() < 0.0 {
            return Err("environment radiance must be finite nonnegative linear RGB".into());
        }
        let mut coefficients = [Vec3::ZERO; 9];
        coefficients[0] = radiance / Y00;
        let environment = Self { coefficients };
        environment.validate()?;
        Ok(environment)
    }
    /// Project tightly packed, top-left-origin linear HDR RGB panorama pixels.
    /// Longitude increases from +X toward +Z; the top row faces +Y.
    /// Uses texel-center directions and exact per-texel solid angle weights.
    pub fn from_equirectangular(
        width: u32,
        height: u32,
        radiance: &[Vec3],
    ) -> Result<Self, BoxError> {
        if width < 4
            || height < 2
            || (width as usize).checked_mul(height as usize) != Some(radiance.len())
        {
            return Err(
                "environment panorama requires width >= 4, height >= 2 and width*height pixels"
                    .into(),
            );
        }
        let mut integral = [[0.0_f64; 3]; 9];
        let pi = std::f64::consts::PI;
        for y in 0..height {
            let theta0 = pi * y as f64 / height as f64;
            let theta1 = pi * (y + 1) as f64 / height as f64;
            let theta = (theta0 + theta1) * 0.5;
            let weight = (theta0.cos() - theta1.cos()) * 2.0 * pi / width as f64;
            for x in 0..width {
                let pixel = radiance[(y as usize) * width as usize + x as usize];
                if !pixel.is_finite() || pixel.min_element() < 0.0 {
                    return Err("environment panorama contains invalid linear HDR radiance".into());
                }
                let phi = 2.0 * pi * (x as f64 + 0.5) / width as f64;
                let direction = Vec3::new(
                    (theta.sin() * phi.cos()) as f32,
                    theta.cos() as f32,
                    (theta.sin() * phi.sin()) as f32,
                );
                for (i, basis) in sh_basis(direction).into_iter().enumerate() {
                    for (channel, value) in pixel.to_array().into_iter().enumerate() {
                        integral[i][channel] += value as f64 * basis as f64 * weight;
                    }
                }
            }
        }
        // Cosine convolution / pi: band 0 = 1, band 1 = 2/3, band 2 = 1/4.
        let coefficients = std::array::from_fn(|i| {
            let convolution = if i == 0 {
                1.0
            } else if i < 4 {
                2.0 / 3.0
            } else {
                0.25
            };
            Vec3::from_array(integral[i].map(|v| (v * convolution) as f32))
        });
        let environment = Self { coefficients };
        environment.validate()?;
        Ok(environment)
    }
    pub fn coefficients(&self) -> &[Vec3; 9] {
        &self.coefficients
    }
    pub fn validate(&self) -> Result<(), BoxError> {
        if self.coefficients.iter().any(|v| !v.is_finite()) {
            return Err("environment SH coefficients exceed finite f32 range".into());
        }
        Ok(())
    }
    /// Evaluate irradiance/pi for a finite unit world normal. Negative SH ringing is clamped.
    pub fn evaluate(&self, normal: Vec3) -> Vec3 {
        self.coefficients
            .iter()
            .zip(sh_basis(normal))
            .map(|(c, b)| *c * b)
            .sum::<Vec3>()
            .max(Vec3::ZERO)
    }
}
fn sh_basis(n: Vec3) -> [f32; 9] {
    [
        Y00,
        0.48860252 * n.y,
        0.48860252 * n.z,
        0.48860252 * n.x,
        1.0925485 * n.x * n.y,
        1.0925485 * n.y * n.z,
        0.31539157 * (3.0 * n.z * n.z - 1.0),
        1.0925485 * n.x * n.z,
        0.54627424 * (n.x * n.x - n.y * n.y),
    ]
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_hdr_environment_keeps_color_and_energy_in_all_directions() {
        let radiance = Vec3::new(0.1, 2.0, 4.0);
        let analytic = DiffuseEnvironment::constant(radiance).unwrap();
        let projected =
            DiffuseEnvironment::from_equirectangular(128, 64, &vec![radiance; 128 * 64]).unwrap();
        for n in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            -Vec3::X,
            -Vec3::Y,
            -Vec3::Z,
            Vec3::ONE.normalize(),
        ] {
            assert!(analytic.evaluate(n).abs_diff_eq(radiance, 1e-5));
            assert!(projected.evaluate(n).abs_diff_eq(radiance, 0.002));
        }
    }
    #[test]
    fn panorama_orientation_preserves_a_bright_upper_hemisphere() {
        let pixels: Vec<_> = (0..64)
            .flat_map(|y| vec![if y < 32 { Vec3::ONE } else { Vec3::ZERO }; 128])
            .collect();
        let env = DiffuseEnvironment::from_equirectangular(128, 64, &pixels).unwrap();
        assert!((env.evaluate(Vec3::Y).x - 1.0).abs() < 0.002);
        assert!(env.evaluate(-Vec3::Y).x < 0.002);
        assert!((env.evaluate(Vec3::X).x - 0.5).abs() < 0.002);
    }
    #[test]
    fn malformed_panorama_and_nonfinite_or_negative_hdr_are_rejected() {
        assert!(DiffuseEnvironment::from_equirectangular(0, 0, &[]).is_err());
        assert!(DiffuseEnvironment::from_equirectangular(4, 2, &[Vec3::ONE; 7]).is_err());
        assert!(DiffuseEnvironment::constant(Vec3::splat(f32::MAX)).is_err());
        assert!(
            DiffuseEnvironment::from_equirectangular(4, 2, &[Vec3::splat(f32::MAX); 8]).is_err()
        );
        for bad in [
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
            Vec3::splat(-0.1),
        ] {
            assert!(DiffuseEnvironment::constant(bad).is_err());
            let mut pixels = [Vec3::ONE; 8];
            pixels[3] = bad;
            assert!(DiffuseEnvironment::from_equirectangular(4, 2, &pixels).is_err());
        }
    }
}

/// Validated scene-linear full-sphere panorama, shared without CPU GGX cooking.
#[derive(Debug, Clone)]
pub struct EnvironmentRadiance {
    width: u32,
    height: u32,
    pixels: std::sync::Arc<[Vec3]>,
}
impl EnvironmentRadiance {
    pub fn new(width: u32, height: u32, pixels: Vec<Vec3>) -> Result<Self, BoxError> {
        if width < 4
            || height < 2
            || (width as usize).checked_mul(height as usize) != Some(pixels.len())
            || pixels
                .iter()
                .any(|v| !v.is_finite() || v.min_element() < 0.0)
        {
            return Err("invalid environment radiance dimensions/pixels".into());
        }
        Ok(Self {
            width,
            height,
            pixels: pixels.into(),
        })
    }
    pub fn constant(color: Vec3) -> Result<Self, BoxError> {
        Self::new(4, 2, vec![color; 8])
    }
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn sample(&self, direction: Vec3) -> Vec3 {
        crate::specular::sample_panorama(self.width, self.height, &self.pixels, direction)
    }
    pub fn pixels(&self) -> &[Vec3] {
        &self.pixels
    }
    pub fn diffuse(&self) -> Result<DiffuseEnvironment, BoxError> {
        DiffuseEnvironment::from_equirectangular(self.width, self.height, &self.pixels)
    }
}

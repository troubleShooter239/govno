//! Small CPU reference cooker for split-sum specular IBL.
use crate::DiffuseEnvironment;
use glam::{Vec2, Vec3};
use govno_core::BoxError;
use std::f32::consts::{PI, TAU};
pub const SPECULAR_WIDTH: u32 = 64;
pub const SPECULAR_HEIGHT: u32 = 32;
pub const SPECULAR_LEVELS: u32 = 6;
pub const BRDF_LUT_SIZE: u32 = 64;
const SAMPLES: u32 = 128;
#[derive(Debug, Clone)]
pub struct SpecularEnvironment {
    diffuse: DiffuseEnvironment,
    source: crate::EnvironmentRadiance,
    atlas: Vec<[f32; 4]>,
}
impl SpecularEnvironment {
    pub fn constant(radiance: Vec3) -> Result<Self, BoxError> {
        Ok(Self {
            diffuse: DiffuseEnvironment::constant(radiance)?,
            source: crate::EnvironmentRadiance::constant(radiance)?,
            atlas: vec![
                radiance.extend(1.0).to_array();
                (SPECULAR_WIDTH * SPECULAR_HEIGHT * SPECULAR_LEVELS) as usize
            ],
        })
    }
    /// Uses the same +Y-up, +X-to-+Z panorama convention as DiffuseEnvironment.
    /// CPU cooking is synchronous; call at load time, not each rendered frame.
    pub fn from_equirectangular(
        width: u32,
        height: u32,
        pixels: &[Vec3],
    ) -> Result<Self, BoxError> {
        let diffuse = DiffuseEnvironment::from_equirectangular(width, height, pixels)?;
        let mut atlas = Vec::new();
        for level in 0..SPECULAR_LEVELS {
            let roughness = level as f32 / (SPECULAR_LEVELS - 1) as f32;
            for y in 0..SPECULAR_HEIGHT {
                for x in 0..SPECULAR_WIDTH {
                    let theta = PI * (y as f32 + 0.5) / SPECULAR_HEIGHT as f32;
                    let phi = TAU * (x as f32 + 0.5) / SPECULAR_WIDTH as f32;
                    let n = Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.cos(),
                        theta.sin() * phi.sin(),
                    );
                    let color = if level == 0 {
                        sample_panorama(width, height, pixels, n)
                    } else {
                        let tangent = (if n.z.abs() < 0.99 { Vec3::Z } else { Vec3::X })
                            .cross(n)
                            .normalize();
                        let bitangent = n.cross(tangent);
                        let mut color = Vec3::ZERO;
                        let mut total = 0.0;
                        for i in 0..SAMPLES {
                            let h = importance_ggx(i, roughness);
                            let h = tangent * h.x + bitangent * h.y + n * h.z;
                            let l = 2.0 * n.dot(h) * h - n;
                            let weight = n.dot(l).max(0.0);
                            if weight > 0.0 {
                                total += weight;
                                color = color.lerp(
                                    sample_panorama(width, height, pixels, l),
                                    weight / total,
                                );
                            }
                        }
                        color
                    };
                    atlas.push(color.extend(1.0).to_array());
                }
            }
        }
        Ok(Self {
            diffuse,
            atlas,
            source: crate::EnvironmentRadiance::new(width, height, pixels.to_vec())?,
        })
    }
    pub fn source(&self) -> &crate::EnvironmentRadiance {
        &self.source
    }
    pub fn diffuse(&self) -> &DiffuseEnvironment {
        &self.diffuse
    }
    pub fn atlas(&self) -> &[[f32; 4]] {
        &self.atlas
    }
}
pub(crate) fn sample_panorama(width: u32, height: u32, pixels: &[Vec3], direction: Vec3) -> Vec3 {
    let u = (direction.z.atan2(direction.x) / TAU).rem_euclid(1.0);
    let v = direction.y.clamp(-1.0, 1.0).acos() / PI;
    let x = u * width as f32 - 0.5;
    let y = (v * height as f32 - 0.5).clamp(0.0, (height - 1) as f32);
    let x0 = x.floor() as i64;
    let y0 = y.floor() as u32;
    let pixel = |x: i64, y: u32| pixels[(y * width) as usize + x.rem_euclid(width as i64) as usize];
    pixel(x0, y0).lerp(pixel(x0 + 1, y0), x - x.floor()).lerp(
        pixel(x0, (y0 + 1).min(height - 1))
            .lerp(pixel(x0 + 1, (y0 + 1).min(height - 1)), x - x.floor()),
        y - y.floor(),
    )
}
fn importance_ggx(i: u32, roughness: f32) -> Vec3 {
    let x = i as f32 / SAMPLES as f32;
    let y = i.reverse_bits() as f64 / 4294967296.0;
    let a2 = roughness.powi(4);
    let cosine = (((1.0 - y) / (1.0 + (a2 as f64 - 1.0) * y)) as f32).sqrt();
    let sine = (1.0 - cosine * cosine).max(0.0).sqrt();
    Vec3::new((TAU * x).cos() * sine, (TAU * x).sin() * sine, cosine)
}
fn integrate_brdf(nv: f32, roughness: f32) -> Vec2 {
    let v = Vec3::new((1.0 - nv * nv).sqrt(), 0.0, nv);
    let a2 = roughness.powi(4);
    let mut result = Vec2::ZERO;
    for i in 0..SAMPLES {
        let h = importance_ggx(i, roughness);
        let vh = v.dot(h).max(0.0);
        let l = 2.0 * vh * h - v;
        let nl = l.z.max(0.0);
        if nl > 0.0 {
            // Height-correlated Smith GGX, matching the Forward direct-light BRDF.
            let visibility = 0.5
                / (nl * (nv * nv * (1.0 - a2) + a2).sqrt()
                    + nv * (nl * nl * (1.0 - a2) + a2).sqrt())
                .max(1e-8);
            let weight = 4.0 * visibility * nl * vh / h.z.max(1e-6);
            let fresnel = (1.0 - vh).powi(5);
            result += Vec2::new(1.0 - fresnel, fresnel) * weight;
        }
    }
    result / SAMPLES as f32
}
/// RG = split-sum scale/bias; x = NdotV, y = perceptual roughness.
pub fn specular_brdf_lut() -> Vec<[f32; 4]> {
    (0..BRDF_LUT_SIZE)
        .flat_map(|y| {
            (0..BRDF_LUT_SIZE).map(move |x| {
                let value = integrate_brdf(
                    (x as f32 + 0.5) / BRDF_LUT_SIZE as f32,
                    (y as f32 + 0.5) / BRDF_LUT_SIZE as f32,
                );
                [value.x, value.y, 0.0, 1.0]
            })
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roughness_filter_broadens_a_bright_panorama_patch() {
        let pixels: Vec<_> = (0..128)
            .map(|i| {
                if i % 16 < 2 {
                    Vec3::splat(4.0)
                } else {
                    Vec3::ZERO
                }
            })
            .collect();
        let env = SpecularEnvironment::from_equirectangular(16, 8, &pixels).unwrap();
        let variance = |level: usize| {
            let plane = (SPECULAR_WIDTH * SPECULAR_HEIGHT) as usize;
            let slice = &env.atlas[level * plane..(level + 1) * plane];
            let mean = slice.iter().map(|p| p[0]).sum::<f32>() / plane as f32;
            slice.iter().map(|p| (p[0] - mean).powi(2)).sum::<f32>() / plane as f32
        };
        assert!(
            env.atlas
                .iter()
                .flatten()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
        assert!(variance(5) < variance(0) * 0.5);
        let lut = specular_brdf_lut();
        assert_eq!(lut.len(), (BRDF_LUT_SIZE * BRDF_LUT_SIZE) as usize);
        assert!(lut.iter().flatten().all(|v| v.is_finite() && *v >= 0.0));
    }
    #[test]
    fn mirror_brdf_matches_analytic_fresnel_and_white_furnace_is_bounded() {
        for nv in [0.01, 0.1, 0.5, 1.0] {
            let value = integrate_brdf(nv, 0.0);
            assert!(value.abs_diff_eq(
                Vec2::new(1.0 - (1.0 - nv).powi(5), (1.0 - nv).powi(5)),
                1e-5
            ));
        }
        for nv in [0.05, 0.5, 1.0] {
            for roughness in [0.2, 0.6, 1.0] {
                let value = integrate_brdf(nv, roughness);
                assert!(
                    value.is_finite() && value.min_element() >= 0.0 && value.element_sum() < 1.05
                );
            }
        }
    }
    #[test]
    fn panorama_filter_preserves_constant_hdr_and_longitude_seam() {
        let color = Vec3::new(2.0, 0.25, 4.0);
        let env = SpecularEnvironment::from_equirectangular(8, 4, &[color; 32]).unwrap();
        assert_eq!(
            env.atlas.len(),
            (SPECULAR_WIDTH * SPECULAR_HEIGHT * SPECULAR_LEVELS) as usize
        );
        assert!(
            env.atlas
                .iter()
                .all(|p| Vec3::new(p[0], p[1], p[2]).abs_diff_eq(color, 1e-5))
        );
        let pixels = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::ONE,
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::ONE,
        ];
        let left = sample_panorama(4, 2, &pixels, Vec3::new(1.0, 0.0, -0.000001));
        let right = sample_panorama(4, 2, &pixels, Vec3::new(1.0, 0.0, 0.000001));
        assert!(left.abs_diff_eq(right, 1e-5));
        assert!(SpecularEnvironment::from_equirectangular(4, 2, &[Vec3::splat(-1.0); 8]).is_err());
    }
}

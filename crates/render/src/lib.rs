//! Renderer-facing CPU data. No Vulkan, window, or ECS dependencies.
mod camera;
mod visibility;
pub use visibility::{Aabb, Frustum};
mod environment;
pub use environment::{DiffuseEnvironment, EnvironmentRadiance};
mod light;
pub use light::{MAX_POINT_LIGHTS, PointLight};
mod post;
mod probe;
pub use probe::{LocalProbe, MAX_LOCAL_PROBES, probe_weights};
mod scene;
mod shadow;
mod specular;
pub use camera::Camera;
pub use post::{AutoExposure, BloomSettings, PostProcessSettings};
pub use scene::{
    DrawInstance, Material, MeshData, MeshHandle, RenderWorld, TextureHandle, Transform, Vertex,
};
pub use shadow::{CASCADE_COUNT, DirectionalShadow, ShadowCascades};
pub use specular::{
    BRDF_LUT_SIZE, SPECULAR_HEIGHT, SPECULAR_LEVELS, SPECULAR_WIDTH, SpecularEnvironment,
    specular_brdf_lut,
};
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderExtent {
    pub width: u32,
    pub height: u32,
}
impl RenderExtent {
    pub fn is_zero(self) -> bool {
        self.width == 0 || self.height == 0
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameStatus {
    Presented,
    Skipped,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TextureColorSpace {
    #[default]
    Srgb,
    Linear,
}
/// Tightly packed, top-left-origin base-color pixels in sRGB RGBA8.
pub struct TextureData {
    pub color_space: TextureColorSpace,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}
impl TextureData {
    pub fn validate(&self) -> Result<(), govno_core::BoxError> {
        let bytes = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|n| n.checked_mul(4));
        if self.width == 0 || self.height == 0 || bytes != Some(self.rgba8.len()) {
            return Err(
                "texture requires positive dimensions and exactly width*height*4 bytes".into(),
            );
        }
        Ok(())
    }
}
#[cfg(test)]
mod texture_tests {
    use super::*;
    #[test]
    fn malformed_pixels_are_rejected() {
        assert!(
            TextureData {
                color_space: Default::default(),
                width: 0,
                height: 1,
                rgba8: vec![]
            }
            .validate()
            .is_err()
        );
        assert!(
            TextureData {
                color_space: Default::default(),
                width: 2,
                height: 2,
                rgba8: vec![255; 15]
            }
            .validate()
            .is_err()
        );
        assert!(
            TextureData {
                color_space: Default::default(),
                width: 1,
                height: 1,
                rgba8: vec![255; 4]
            }
            .validate()
            .is_ok()
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MipFilter {
    None,
    Nearest,
    Linear,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureWrap {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}
#[derive(Debug, Clone, Copy)]
pub struct TextureSampling {
    pub min: TextureFilter,
    pub mag: TextureFilter,
    pub mip: MipFilter,
    pub wrap_u: TextureWrap,
    pub wrap_v: TextureWrap,
}
impl Default for TextureSampling {
    fn default() -> Self {
        Self {
            min: TextureFilter::Linear,
            mag: TextureFilter::Linear,
            mip: MipFilter::Linear,
            wrap_u: TextureWrap::Repeat,
            wrap_v: TextureWrap::Repeat,
        }
    }
}
impl TextureSampling {
    pub fn mip_levels(self, width: u32, height: u32) -> u32 {
        if width == 0 || height == 0 {
            return 0;
        }
        if self.mip == MipFilter::None {
            1
        } else {
            u32::BITS - width.max(height).leading_zeros()
        }
    }
}
#[cfg(test)]
mod mip_tests {
    use super::*;
    #[test]
    fn mip_count_handles_rectangular_and_non_power_of_two_images() {
        let sampling = TextureSampling::default();
        assert_eq!(sampling.mip_levels(1, 1), 1);
        assert_eq!(sampling.mip_levels(64, 8), 7);
        assert_eq!(sampling.mip_levels(7, 3), 3);
        assert_eq!(sampling.mip_levels(0, 4), 0);
        assert_eq!(
            TextureSampling {
                mip: MipFilter::None,
                ..sampling
            }
            .mip_levels(64, 64),
            1
        );
    }
}

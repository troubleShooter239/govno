//! Renderer-facing CPU data. No Vulkan, window, or ECS dependencies.
mod camera;
mod scene;
pub use camera::Camera;
pub use scene::{
    DrawInstance, Material, MeshData, MeshHandle, RenderWorld, TextureHandle, Transform, Vertex,
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

/// Tightly packed, top-left-origin base-color pixels in sRGB RGBA8.
pub struct TextureData {
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
                width: 0,
                height: 1,
                rgba8: vec![]
            }
            .validate()
            .is_err()
        );
        assert!(
            TextureData {
                width: 2,
                height: 2,
                rgba8: vec![255; 15]
            }
            .validate()
            .is_err()
        );
        assert!(
            TextureData {
                width: 1,
                height: 1,
                rgba8: vec![255; 4]
            }
            .validate()
            .is_ok()
        );
    }
}

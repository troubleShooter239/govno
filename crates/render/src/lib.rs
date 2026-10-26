//! Renderer-facing CPU data. No Vulkan, window, or ECS dependencies.
mod camera;
mod scene;
pub use camera::Camera;
pub use scene::{DrawInstance, Material, MeshData, MeshHandle, RenderWorld, Transform, Vertex};
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

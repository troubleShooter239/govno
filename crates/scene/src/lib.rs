//! Versioned authored scenes and shared CPU model preparation.
//! No Vulkan, window or ECS dependency. Instantiation is owned by the application.
mod description;
mod prepared;
pub use description::{
    CameraDescription, MAX_SCENE_BYTES, MAX_SCENE_MODELS, ModelDescription, PhysicsDescription,
    PointLightDescription, SceneDescription, SunDescription, TransformDescription,
};
pub use prepared::{ModelAssets, PreparedModel, PreparedScene};

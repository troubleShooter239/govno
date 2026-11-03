use glam::{Quat, Vec3};
use govno_core::BoxError;
use govno_render::{Camera, PointLight, RenderExtent, Transform};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs::File, io::Read, path::Path};

pub const MAX_SCENE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_SCENE_MODELS: usize = 10_000;

/// Version 1 is deliberately flat: model placements, an initial camera and lights.
/// Node hierarchies, animation and editor metadata are outside this MVP format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDescription {
    pub version: u32,
    pub name: String,
    #[serde(default)]
    pub camera: CameraDescription,
    #[serde(default)]
    pub sun: SunDescription,
    #[serde(default)]
    pub point_lights: Vec<PointLightDescription>,
    #[serde(default)]
    pub models: Vec<ModelDescription>,
    #[serde(default)]
    pub player: bool,
}

impl SceneDescription {
    pub fn from_json(bytes: &[u8]) -> Result<Self, BoxError> {
        if bytes.len() > MAX_SCENE_BYTES {
            return Err("scene JSON exceeds 4 MiB".into());
        }
        let description: Self = serde_json::from_slice(bytes)?;
        description.validate()?;
        Ok(description)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, BoxError> {
        // A bounded read also handles files that grow while being read.
        let mut bytes = Vec::new();
        File::open(path)?
            .take(MAX_SCENE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        Self::from_json(&bytes)
    }

    pub fn to_json(&self) -> Result<String, BoxError> {
        self.validate()?;
        let json = serde_json::to_string_pretty(self)?;
        if json.len() > MAX_SCENE_BYTES {
            return Err("serialized scene JSON exceeds 4 MiB".into());
        }
        Ok(json)
    }

    pub fn validate(&self) -> Result<(), BoxError> {
        if self.version != 1 {
            return Err(format!("unsupported scene version: {}", self.version).into());
        }
        if self.name.trim().is_empty() || self.name.len() > 256 {
            return Err("scene name must contain 1..=256 bytes".into());
        }
        self.camera.to_camera().view_projection(RenderExtent {
            width: 1,
            height: 1,
        })?;
        let direction = Vec3::from_array(self.sun.direction);
        let radiance = Vec3::from_array(self.sun.radiance);
        if !direction.is_finite()
            || direction.length_squared() < 1e-8
            || !direction.length_squared().is_finite()
            || !radiance.is_finite()
            || radiance.min_element() < 0.0
        {
            return Err("scene sun needs finite nonzero direction and nonnegative radiance".into());
        }
        if self.point_lights.len() > govno_render::MAX_POINT_LIGHTS {
            return Err("scene exceeds the Forward renderer's 16 point light limit".into());
        }
        for light in &self.point_lights {
            light.to_light().validate()?;
        }
        if self.models.len() > MAX_SCENE_MODELS {
            return Err("scene exceeds 10000 model placements".into());
        }
        let mut ids = HashSet::new();
        for model in &self.models {
            if model.id.trim().is_empty() || model.id.len() > 256 || !ids.insert(&model.id) {
                return Err(format!("invalid or duplicate scene model ID: {:?}", model.id).into());
            }
            // Portable, scene-relative paths. Parent segments are allowed for
            // references into sibling asset directories; this is not a filesystem sandbox.
            if model.asset.is_empty()
                || model.asset.len() > 4096
                || model.asset.starts_with('/')
                || model.asset.contains(['\\', ':', '\0'])
                || model.asset.split('/').any(str::is_empty)
            {
                return Err(
                    format!("model {} needs a portable relative asset path", model.id).into(),
                );
            }
            model.transform.to_transform().validate()?;
            if let Some(body) = &model.physics {
                let half =
                    Vec3::from_array(body.half_extents) * Vec3::from_array(model.transform.scale);
                if !half.is_finite() || half.min_element() < 0.001 || half.max_element() > 10000.0 {
                    return Err(format!(
                        "model {} needs collider half extents in [0.001,10000]",
                        model.id
                    )
                    .into());
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDescription {
    /// Stable authored identity, preserved across save/load; not an ECS entity ID.
    pub id: String,
    /// UTF-8 path with forward slashes, relative to the scene JSON file.
    pub asset: String,
    #[serde(default)]
    pub transform: TransformDescription,
    #[serde(default)]
    pub physics: Option<PhysicsDescription>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicsDescription {
    /// Local cuboid half extents, scaled by the model transform.
    pub half_extents: [f32; 3],
    #[serde(default)]
    pub dynamic: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TransformDescription {
    pub translation: [f32; 3],
    /// Quaternion in xyzw order.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}
impl Default for TransformDescription {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}
impl TransformDescription {
    pub fn to_transform(&self) -> Transform {
        Transform {
            translation: Vec3::from_array(self.translation),
            rotation: Quat::from_array(self.rotation),
            scale: Vec3::from_array(self.scale),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraDescription {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub vertical_fov: f32,
    pub near: f32,
}
impl Default for CameraDescription {
    fn default() -> Self {
        let c = Camera::default();
        Self {
            position: c.position.to_array(),
            yaw: c.yaw,
            pitch: c.pitch,
            vertical_fov: c.vertical_fov,
            near: c.near,
        }
    }
}
impl CameraDescription {
    pub fn to_camera(&self) -> Camera {
        Camera {
            position: Vec3::from_array(self.position),
            yaw: self.yaw,
            pitch: self.pitch,
            vertical_fov: self.vertical_fov,
            near: self.near,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SunDescription {
    /// Direction from the surface toward the light.
    pub direction: [f32; 3],
    pub radiance: [f32; 3],
}
impl Default for SunDescription {
    fn default() -> Self {
        Self {
            direction: [-0.5, 1.0, 0.7],
            radiance: [4.0; 3],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointLightDescription {
    pub position: [f32; 3],
    pub intensity: [f32; 3],
    pub range: f32,
}
impl PointLightDescription {
    pub fn to_light(&self) -> PointLight {
        PointLight {
            position: Vec3::from_array(self.position),
            intensity: Vec3::from_array(self.intensity),
            range: self.range,
        }
    }
}

use crate::{Camera, RenderExtent};
use glam::{Mat4, Quat, Vec3};
use govno_core::BoxError;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// xyz tangent, w handedness (+/-1); all zero means use derivative fallback.
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
    pub position: [f32; 3],
    pub normal: [f32; 3],
}
pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}
impl MeshData {
    pub fn validate(&self) -> Result<(), BoxError> {
        if self.vertices.is_empty()
            || self.indices.is_empty()
            || !self.indices.len().is_multiple_of(3)
            || self.vertices.len() > u32::MAX as usize
            || self.indices.len() > u32::MAX as usize
        {
            return Err("mesh must contain indexed triangles within u32 limits".into());
        }
        for vertex in &self.vertices {
            let tangent = Vec3::new(vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]);
            if !vertex.tangent.iter().all(|v| v.is_finite())
                || (vertex.tangent[3] != 0.0
                    && (vertex.tangent[3].abs() != 1.0 || tangent.length_squared() < 1e-8))
            {
                return Err("invalid vertex tangent or handedness".into());
            }
            if !vertex.uv.iter().all(|v| v.is_finite())
                || !Vec3::from_array(vertex.position).is_finite()
                || !Vec3::from_array(vertex.normal).is_finite()
                || Vec3::from_array(vertex.normal).length_squared() < 1e-8
            {
                return Err("mesh contains non-finite attributes or a zero normal".into());
            }
        }
        if self
            .indices
            .iter()
            .any(|&i| i as usize >= self.vertices.len())
        {
            return Err("mesh index out of bounds".into());
        }
        Ok(())
    }
}
/// Renderer-scoped, monotonically assigned ID. Never a raw GPU handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshHandle {
    owner: u64,
    index: u64,
}
impl MeshHandle {
    /// Backend constructor. A forged, foreign or removed handle is rejected at draw time.
    pub fn from_parts(owner: u64, index: u64) -> Self {
        Self { owner, index }
    }
}
#[derive(Debug, Clone)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}
impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}
impl Transform {
    pub fn validate(&self) -> Result<(), BoxError> {
        if !self.translation.is_finite()
            || !self.rotation.is_finite()
            || (self.rotation.length_squared() - 1.0).abs() > 0.001
            || !self.scale.is_finite()
            || self.scale.min_element() < 0.0001
            || !(self.scale * self.scale).is_finite()
            || !(self.scale * self.scale).recip().is_finite()
            || !self.matrix().is_finite()
        {
            return Err(
                "transform needs finite TRS, unit quaternion and positive nonzero scale".into(),
            );
        }
        Ok(())
    }
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub emissive_texture: Option<TextureHandle>,
    pub occlusion_texture: Option<TextureHandle>,
    pub emissive: Vec3,
    pub occlusion_strength: f32,
    pub normal_texture: Option<TextureHandle>,
    pub metallic_roughness_texture: Option<TextureHandle>,
    pub normal_scale: f32,
    pub base_color_texture: Option<TextureHandle>,
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
}
impl Material {
    pub fn validate(self) -> Result<(), BoxError> {
        if !self.emissive.is_finite()
            || self.emissive.min_element() < 0.0
            || !self.occlusion_strength.is_finite()
            || !(0.0..=1.0).contains(&self.occlusion_strength)
            || !self.normal_scale.is_finite()
            || self.normal_scale < 0.0
            || !self.base_color.is_finite()
            || self.base_color.min_element() < 0.0
            || self.base_color.max_element() > 1.0
            || !self.metallic.is_finite()
            || !(0.0..=1.0).contains(&self.metallic)
            || !self.roughness.is_finite()
            || !(0.045..=1.0).contains(&self.roughness)
        {
            return Err(
                "material needs linear RGB/metallic in [0,1], roughness in [0.045,1]".into(),
            );
        }
        Ok(())
    }
}
pub struct DrawInstance {
    pub mesh: MeshHandle,
    pub transform: Transform,
    pub material: Material,
}
pub struct RenderWorld {
    pub frustum_culling: bool,
    pub post_process: crate::PostProcessSettings,
    pub directional_shadow: Option<crate::DirectionalShadow>,
    /// None disables diffuse IBL. Coefficients are a CPU fallback; an uploaded
    /// global GPU environment overrides them with its computed SH.
    pub diffuse_environment: Option<crate::DiffuseEnvironment>,
    pub specular_environment: bool,
    pub local_probes: Vec<crate::LocalProbe>,
    pub camera: Camera,
    /// Direction from a surface toward the light, not the direction of propagation.
    pub light_direction: Vec3,
    pub light_radiance: Vec3,
    pub point_lights: Vec<crate::PointLight>,
    pub instances: Vec<DrawInstance>,
}
impl RenderWorld {
    pub fn validate(&self, extent: RenderExtent) -> Result<(), BoxError> {
        self.post_process.validate()?;
        if self.local_probes.len() > crate::MAX_LOCAL_PROBES {
            return Err("too many local probes".into());
        }
        if let Some(environment) = &self.diffuse_environment {
            environment.validate()?;
        }
        if let Some(shadow) = self.directional_shadow {
            shadow.cascades(&self.camera, extent, self.light_direction)?;
        }
        for (i, probe) in self.local_probes.iter().enumerate() {
            probe.validate()?;
            if self.local_probes[..i].iter().any(|p| p.slot == probe.slot) {
                return Err("duplicate local probe slot".into());
            }
        }
        self.camera.view_projection(extent)?;
        if !self.light_direction.is_finite()
            || self.light_direction.length_squared() < 1e-8
            || !self.light_radiance.is_finite()
            || self.light_radiance.min_element() < 0.0
        {
            return Err("invalid directional light".into());
        }
        if self.point_lights.len() > crate::MAX_POINT_LIGHTS {
            return Err("too many point lights for the basic Forward renderer (maximum 16)".into());
        }
        for light in &self.point_lights {
            light.validate()?;
        }
        for instance in &self.instances {
            instance.transform.validate()?;
            instance.material.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureHandle {
    owner: u64,
    index: u64,
}
impl TextureHandle {
    pub fn from_parts(owner: u64, index: u64) -> Self {
        Self { owner, index }
    }
}

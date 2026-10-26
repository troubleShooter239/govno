use crate::{Camera, RenderExtent};
use glam::{Mat4, Quat, Vec3};
use govno_core::BoxError;

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
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
    pub base_color_texture: Option<TextureHandle>,
    pub base_color: Vec3,
    pub metallic: f32,
    pub roughness: f32,
}
impl Material {
    pub fn validate(self) -> Result<(), BoxError> {
        if !self.base_color.is_finite()
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
    pub camera: Camera,
    /// Direction from a surface toward the light, not the direction of propagation.
    pub light_direction: Vec3,
    pub light_radiance: Vec3,
    pub instances: Vec<DrawInstance>,
}
impl RenderWorld {
    pub fn validate(&self, extent: RenderExtent) -> Result<(), BoxError> {
        self.camera.view_projection(extent)?;
        if !self.light_direction.is_finite()
            || self.light_direction.length_squared() < 1e-8
            || !self.light_radiance.is_finite()
            || self.light_radiance.min_element() < 0.0
        {
            return Err("invalid directional light".into());
        }
        for instance in &self.instances {
            instance.transform.validate()?;
            instance.material.validate()?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bad_mesh_is_rejected_before_gpu_upload() {
        let mut mesh = MeshData {
            vertices: vec![Vertex {
                position: [0.0; 3],
                uv: [0.0; 2],
                normal: [0.0, 1.0, 0.0],
            }],
            indices: vec![0, 0, 1],
        };
        assert!(mesh.validate().is_err());
        mesh.indices[2] = 0;
        assert!(mesh.validate().is_ok());
        mesh.vertices[0].normal = [0.0; 3];
        assert!(mesh.validate().is_err());
    }
    #[test]
    fn singular_transform_and_invalid_material_are_rejected() {
        assert!(
            Transform {
                scale: Vec3::ZERO,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Material {
                base_color_texture: None,
                base_color: Vec3::ONE,
                metallic: f32::NAN,
                roughness: 0.5
            }
            .validate()
            .is_err()
        );
    }
    #[test]
    fn scaled_normal_matches_inverse_transpose() {
        let t = Transform {
            scale: Vec3::new(2.0, 0.5, 3.0),
            rotation: Quat::from_rotation_y(0.6),
            ..Default::default()
        };
        let normal = Vec3::new(1.0, 1.0, 0.0).normalize();
        let expected = t
            .matrix()
            .inverse()
            .transpose()
            .transform_vector3(normal)
            .normalize();
        let actual = t
            .matrix()
            .transform_vector3(normal / (t.scale * t.scale))
            .normalize();
        assert!(actual.abs_diff_eq(expected, 1e-5));
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

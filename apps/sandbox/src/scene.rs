use glam::{Quat, Vec3};
use govno_core::BoxError;
use govno_render::{
    Camera, DrawInstance, Material, MeshData, MeshHandle, RenderWorld, Transform, Vertex,
};
use govno_render_vulkan::VulkanRenderer;

pub fn cube() -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (normal, tangent) in [
        (Vec3::X, Vec3::Z),
        (-Vec3::X, Vec3::Z),
        (Vec3::Y, Vec3::X),
        (-Vec3::Y, Vec3::X),
        (Vec3::Z, Vec3::X),
        (-Vec3::Z, Vec3::X),
    ] {
        let bitangent = normal.cross(tangent);
        let start = vertices.len() as u32;
        for (u, v) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            vertices.push(Vertex {
                tangent: [0.0; 4],
                position: ((normal + tangent * u + bitangent * v) * 0.5).to_array(),
                normal: normal.to_array(),
                uv: [(u + 1.0) * 0.5, (v + 1.0) * 0.5],
            });
        }
        indices.extend([start, start + 1, start + 2, start, start + 2, start + 3]);
    }
    MeshData { vertices, indices }
}
pub fn sphere() -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    const RINGS: u32 = 24;
    const SEGMENTS: u32 = 32;
    for ring in 0..=RINGS {
        let theta = ring as f32 / RINGS as f32 * std::f32::consts::PI;
        for segment in 0..=SEGMENTS {
            let phi = segment as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            let normal = Vec3::new(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );
            vertices.push(Vertex {
                tangent: [0.0; 4],
                position: normal.to_array(),
                normal: normal.to_array(),
                uv: [segment as f32 / SEGMENTS as f32, ring as f32 / RINGS as f32],
            });
        }
    }
    for ring in 0..RINGS {
        for segment in 0..SEGMENTS {
            let a = ring * (SEGMENTS + 1) + segment;
            let b = a + SEGMENTS + 1;
            indices.extend([a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    MeshData { vertices, indices }
}
pub fn build(
    renderer: &mut VulkanRenderer,
    smoke: bool,
) -> Result<(RenderWorld, usize, Option<MeshHandle>), BoxError> {
    let mut rgba8 = Vec::new();
    for y in 0..64 {
        for x in 0..64 {
            let value = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 155 };
            rgba8.extend_from_slice(&[value, value, value, 255]);
        }
    }
    renderer.set_base_color_texture(&govno_render::TextureData {
        color_space: Default::default(),
        width: 64,
        height: 64,
        rgba8,
    })?;
    let cube = renderer.upload_mesh(&cube())?;
    let sphere = renderer.upload_mesh(&sphere())?;
    let white = renderer.upload_texture(&govno_render::TextureData {
        color_space: Default::default(),
        width: 1,
        height: 1,
        rgba8: vec![255; 4],
    })?;
    let mut instances = Vec::new();
    for row in 0..2 {
        for column in 0..5 {
            instances.push(DrawInstance {
                mesh: sphere,
                transform: Transform {
                    translation: Vec3::new((column as f32 - 2.0) * 1.9, 0.0, -row as f32 * 2.5),
                    scale: Vec3::splat(0.8),
                    ..Default::default()
                },
                material: Material {
                    emissive_texture: None,
                    occlusion_texture: None,
                    emissive: Vec3::ZERO,
                    occlusion_strength: 1.0,
                    normal_texture: None,
                    metallic_roughness_texture: None,
                    normal_scale: 1.0,
                    base_color_texture: Some(white),
                    base_color: if row == 0 {
                        Vec3::new(0.12, 0.48, 0.8)
                    } else {
                        Vec3::new(0.95, 0.55, 0.22)
                    },
                    metallic: row as f32,
                    roughness: 0.12 + column as f32 * 0.2,
                },
            });
        }
    }
    let animated = instances.len();
    instances.push(DrawInstance {
        mesh: cube,
        transform: Transform {
            translation: Vec3::new(0.0, 0.1, 2.2),
            scale: Vec3::new(1.5, 1.8, 0.9),
            rotation: Quat::from_rotation_y(0.3),
        },
        material: Material {
            emissive_texture: None,
            occlusion_texture: None,
            emissive: Vec3::ZERO,
            occlusion_strength: 1.0,
            normal_texture: None,
            metallic_roughness_texture: None,
            normal_scale: 1.0,
            base_color_texture: Some(white),
            base_color: Vec3::new(0.65, 0.12, 0.08),
            metallic: 0.1,
            roughness: 0.4,
        },
    });
    // Floor deliberately draws after objects: it must not overwrite closer surfaces.
    instances.push(DrawInstance {
        mesh: cube,
        transform: Transform {
            translation: Vec3::new(0.0, -1.0, 0.0),
            scale: Vec3::new(14.0, 0.4, 14.0),
            ..Default::default()
        },
        material: Material {
            emissive_texture: None,
            occlusion_texture: None,
            emissive: Vec3::ZERO,
            occlusion_strength: 1.0,
            normal_texture: None,
            metallic_roughness_texture: None,
            normal_scale: 1.0,
            base_color_texture: None,
            base_color: Vec3::splat(0.24),
            metallic: 0.0,
            roughness: 0.9,
        },
    });
    let mut normals = Vec::new();
    let mut metal_rough = Vec::new();
    for y in 0..32 {
        for x in 0..32 {
            let nx = ((x as f32 * 0.65).sin() * 0.35 * 127.0 + 128.0) as u8;
            let ny = ((y as f32 * 0.65).cos() * 0.35 * 127.0 + 128.0) as u8;
            normals.extend_from_slice(&[nx, ny, 240, 255]);
            metal_rough.extend_from_slice(&[255, (40 + x * 6) as u8, 220, 255]);
        }
    }
    instances[animated].material.normal_texture =
        Some(renderer.upload_texture(&govno_render::TextureData {
            width: 32,
            height: 32,
            rgba8: normals,
            color_space: govno_render::TextureColorSpace::Linear,
        })?);
    instances[animated].material.metallic_roughness_texture =
        Some(renderer.upload_texture(&govno_render::TextureData {
            width: 32,
            height: 32,
            rgba8: metal_rough,
            color_space: govno_render::TextureColorSpace::Linear,
        })?);
    instances[animated].material.metallic = 0.8;
    instances[animated].material.roughness = 0.9;
    instances[animated].material.emissive = Vec3::new(0.2, 0.025, 0.01);
    let temporary = if smoke {
        let handle = renderer.upload_mesh(&self::cube())?;
        instances.push(DrawInstance {
            mesh: handle,
            transform: Transform {
                translation: Vec3::new(5.0, 0.0, 1.0),
                ..Default::default()
            },
            material: Material {
                emissive_texture: None,
                occlusion_texture: None,
                emissive: Vec3::ZERO,
                occlusion_strength: 1.0,
                normal_texture: None,
                metallic_roughness_texture: None,
                normal_scale: 1.0,
                base_color_texture: None,
                base_color: Vec3::new(0.1, 0.8, 0.2),
                metallic: 0.0,
                roughness: 0.4,
            },
        });
        Some(handle)
    } else {
        None
    };
    let world = RenderWorld {
        post_process: Default::default(),
        camera: Camera {
            position: Vec3::new(0.0, 3.3, 10.0),
            pitch: -0.23,
            ..Default::default()
        },
        light_direction: Vec3::new(-0.5, 1.0, 0.7),
        light_radiance: Vec3::splat(4.0),
        instances,
    };
    Ok((world, animated, temporary))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sample_geometry_has_valid_indices_and_normals() {
        for mesh in [cube(), sphere()] {
            mesh.validate().unwrap();
            for vertex in mesh.vertices {
                assert!((Vec3::from_array(vertex.normal).length() - 1.0).abs() < 1e-5);
            }
        }
    }
}

#[cfg(test)]
mod winding_tests {
    use super::*;
    #[test]
    fn visible_triangles_wind_outwards() {
        for mesh in [cube(), sphere()] {
            for triangle in mesh.indices.as_chunks::<3>().0 {
                let a = &mesh.vertices[triangle[0] as usize];
                let b = &mesh.vertices[triangle[1] as usize];
                let c = &mesh.vertices[triangle[2] as usize];
                let normal = (Vec3::from_array(b.position) - Vec3::from_array(a.position))
                    .cross(Vec3::from_array(c.position) - Vec3::from_array(a.position));
                if normal.length_squared() > 1e-10 {
                    assert!(normal.dot(Vec3::from_array(a.normal)) > 0.0);
                }
            }
        }
    }
}

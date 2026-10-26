//! CPU-only static glTF/GLB decoding. No Vulkan or window dependencies.
use glam::{Mat4, Vec3};
use govno_core::BoxError;
use govno_render::{Material, MeshData, TextureData, Vertex};
use std::path::Path;

pub struct ImportedPrimitive {
    pub mesh: MeshData,
    pub material: Material,
    pub base_color_texture: Option<TextureData>,
    pub sampling: govno_render::TextureSampling,
}
pub fn load_gltf(path: impl AsRef<Path>) -> Result<Vec<ImportedPrimitive>, BoxError> {
    let (document, buffers, images) = gltf::import(path)?;
    decode(document, buffers, images)
}
fn decode(
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
    images: Vec<gltf::image::Data>,
) -> Result<Vec<ImportedPrimitive>, BoxError> {
    if document.extensions_required().next().is_some() || document.animations().next().is_some() {
        return Err(
            "required extensions and animations are not supported by the static importer".into(),
        );
    }
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or("glTF has no scene")?;
    let mut result = Vec::new();
    for node in scene.nodes() {
        visit(node, Mat4::IDENTITY, 0, &buffers, &images, &mut result)?;
    }
    if result.is_empty() {
        return Err("glTF scene has no supported mesh primitives".into());
    }
    Ok(result)
}
fn visit(
    node: gltf::Node<'_>,
    parent: Mat4,
    depth: usize,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    output: &mut Vec<ImportedPrimitive>,
) -> Result<(), BoxError> {
    if depth > 256 || node.skin().is_some() {
        return Err("node hierarchy too deep or skinned".into());
    }
    let matrix = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
    if !matrix.is_finite() || matrix.determinant().abs() < 1e-10 {
        return Err("singular/non-finite glTF transform".into());
    }
    let normals_matrix = matrix.inverse().transpose();
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles
                || primitive.morph_targets().next().is_some()
            {
                return Err("only static TRIANGLES primitives are supported".into());
            }
            let source = primitive.material();
            let pbr = source.pbr_metallic_roughness();
            if source.alpha_mode() != gltf::material::AlphaMode::Opaque
                || source.normal_texture().is_some()
                || source.occlusion_texture().is_some()
                || source.emissive_texture().is_some()
                || source.emissive_factor() != [0.0; 3]
                || pbr.metallic_roughness_texture().is_some()
            {
                return Err(
                    "unsupported material: need opaque base-color-only metallic/roughness".into(),
                );
            }
            let factor = pbr.base_color_factor();
            let material = Material {
                base_color_texture: None,
                base_color: Vec3::new(factor[0], factor[1], factor[2]),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor().max(0.045),
            };
            material.validate()?;
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));
            let positions: Vec<_> = reader
                .read_positions()
                .ok_or("mesh is missing POSITION")?
                .collect();
            let normals: Vec<_> = reader
                .read_normals()
                .ok_or("mesh is missing NORMAL")?
                .collect();
            let uv: Option<Vec<_>> = reader
                .read_tex_coords(0)
                .map(|coords| coords.into_f32().collect());
            if normals.len() != positions.len()
                || uv.as_ref().is_some_and(|uv| uv.len() != positions.len())
            {
                return Err("attribute counts differ".into());
            }
            let mut sampling = govno_render::TextureSampling::default();
            let texture = if let Some(info) = pbr.base_color_texture() {
                if info.tex_coord() != 0 || uv.is_none() {
                    return Err("base-color texture requires TEXCOORD_0".into());
                }
                let sampler = info.texture().sampler();
                sampling = decode_sampler(sampler);
                let image = &images[info.texture().source().index()];
                let rgba8 = match image.format {
                    gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
                    gltf::image::Format::R8G8B8 => image
                        .pixels
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .flat_map(|p| [p[0], p[1], p[2], 255])
                        .collect(),
                    _ => return Err("only RGB8/RGBA8 images are supported".into()),
                };
                let data = TextureData {
                    width: image.width,
                    height: image.height,
                    rgba8,
                };
                data.validate()?;
                Some(data)
            } else {
                None
            };
            let vertices = positions
                .iter()
                .zip(&normals)
                .enumerate()
                .map(|(i, (position, normal))| Vertex {
                    position: matrix
                        .transform_point3(Vec3::from_array(*position))
                        .to_array(),
                    normal: normals_matrix
                        .transform_vector3(Vec3::from_array(*normal))
                        .normalize_or_zero()
                        .to_array(),
                    uv: uv.as_ref().map_or([0.0; 2], |uv| uv[i]),
                })
                .collect();
            let mut indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            if matrix.determinant() < 0.0 {
                for triangle in indices.as_chunks_mut::<3>().0 {
                    triangle.swap(1, 2);
                }
            }
            let mesh = MeshData { vertices, indices };
            mesh.validate()?;
            output.push(ImportedPrimitive {
                mesh,
                material,
                base_color_texture: texture,
                sampling,
            });
        }
    }
    for child in node.children() {
        visit(child, matrix, depth + 1, buffers, images, output)?;
    }
    Ok(())
}
fn decode_sampler(sampler: gltf::texture::Sampler<'_>) -> govno_render::TextureSampling {
    use gltf::texture::{MagFilter, MinFilter, WrappingMode};
    use govno_render::{MipFilter as M, TextureFilter as F, TextureWrap as W};
    let (min, mip) = match sampler.min_filter() {
        Some(MinFilter::Nearest) => (F::Nearest, M::None),
        Some(MinFilter::Linear) => (F::Linear, M::None),
        Some(MinFilter::NearestMipmapNearest) => (F::Nearest, M::Nearest),
        Some(MinFilter::LinearMipmapNearest) => (F::Linear, M::Nearest),
        Some(MinFilter::NearestMipmapLinear) => (F::Nearest, M::Linear),
        Some(MinFilter::LinearMipmapLinear) | None => (F::Linear, M::Linear),
    };
    let wrap = |value| match value {
        WrappingMode::Repeat => W::Repeat,
        WrappingMode::MirroredRepeat => W::MirroredRepeat,
        WrappingMode::ClampToEdge => W::ClampToEdge,
    };
    govno_render::TextureSampling {
        min,
        mip,
        mag: if sampler.mag_filter() == Some(MagFilter::Nearest) {
            F::Nearest
        } else {
            F::Linear
        },
        wrap_u: wrap(sampler.wrap_s()),
        wrap_v: wrap(sampler.wrap_t()),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn modified(source: &str) -> Result<Vec<ImportedPrimitive>, BoxError> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/triangle.gltf");
        let (_, buffers, images) = gltf::import(path)?;
        let document = gltf::Gltf::from_slice(source.as_bytes())?.document;
        decode(document, buffers, images)
    }
    #[test]
    fn explicit_sampler_filters_and_address_modes_survive_import() {
        let source = include_str!("../tests/fixtures/triangle.gltf")
            .replace("\"textures\":[", "\"samplers\":[{\"minFilter\":9987,\"magFilter\":9728,\"wrapS\":33648,\"wrapT\":33071}],\"textures\":[")
            .replace("\"source\":0", "\"source\":0,\"sampler\":0");
        let item = modified(&source).unwrap().remove(0);
        assert_eq!(item.sampling.mag, govno_render::TextureFilter::Nearest);
        assert_eq!(item.sampling.mip, govno_render::MipFilter::Linear);
        assert_eq!(
            item.sampling.wrap_u,
            govno_render::TextureWrap::MirroredRepeat
        );
        assert_eq!(item.sampling.wrap_v, govno_render::TextureWrap::ClampToEdge);
    }
    #[test]
    fn nested_transforms_are_accumulated() {
        let source = include_str!("../tests/fixtures/triangle.gltf")
            .replace(
                "\"children\":[1]",
                "\"children\":[1],\"translation\":[2,0,0]",
            )
            .replace("\"mesh\":0", "\"mesh\":0,\"translation\":[1,0,0]");
        let items = modified(&source).unwrap();
        assert_eq!(items[0].mesh.vertices[0].position, [2.0, 0.0, 0.0]);
    }
    #[test]
    fn mirrored_node_corrects_triangle_winding() {
        let source = include_str!("../tests/fixtures/triangle.gltf")
            .replace("\"mesh\":0", "\"mesh\":0,\"scale\":[-1,1,1]");
        assert_eq!(modified(&source).unwrap()[0].mesh.indices, [0, 2, 1]);
    }
    #[test]
    fn unsupported_primitive_is_rejected() {
        let source = include_str!("../tests/fixtures/triangle.gltf")
            .replace("\"indices\":3", "\"indices\":3,\"mode\":1");
        assert!(modified(&source).is_err());
    }
    #[test]
    fn fixture_decodes_hierarchy_material_and_pixels() {
        let items = load_gltf(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/triangle.gltf"
        ))
        .unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].mesh.indices, [0, 1, 2]);
        assert_eq!(items[0].mesh.vertices[0].position, [-1.0, 0.0, 0.0]);
        assert_eq!(
            items[0].base_color_texture.as_ref().unwrap().rgba8.len(),
            16
        );
    }
}

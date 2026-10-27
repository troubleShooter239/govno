//! CPU-only static glTF/GLB and Radiance HDR decoding. No Vulkan or window dependencies.
mod exr;
mod hdr;
pub use exr::{decode_exr, load_environment};
use glam::{Mat4, Vec3};
use govno_core::BoxError;
use govno_render::{Material, MeshData, TextureData, Vertex};
pub use hdr::{HdrPanorama, MAX_HDR_PIXELS, decode_hdr, load_hdr};
use std::path::Path;

pub struct ImportedPrimitive {
    pub mesh: MeshData,
    pub material: Material,
    pub base_color_texture: Option<TextureData>,
    pub sampling: govno_render::TextureSampling,
    pub emissive_texture: Option<(TextureData, govno_render::TextureSampling)>,
    pub occlusion_texture: Option<(TextureData, govno_render::TextureSampling)>,
    pub normal_texture: Option<(TextureData, govno_render::TextureSampling)>,
    pub metallic_roughness_texture: Option<(TextureData, govno_render::TextureSampling)>,
}
pub fn load_gltf(path: impl AsRef<Path>) -> Result<Vec<ImportedPrimitive>, BoxError> {
    let path = path.as_ref();
    let gltf::Gltf { document, blob } = gltf::Gltf::open(path)?;
    let base = path.parent();
    let buffers = gltf::import_buffers(&document, base, blob)?;
    validate_buffer_views(&document, &buffers)?;
    let images = gltf::import_images(&document, base, &buffers)?;
    decode(document, buffers, images)
}
// The dependency slices image views directly. Check all ranges before decoding.
fn validate_buffer_views(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
) -> Result<(), BoxError> {
    for view in document.views() {
        let buffer = buffers
            .get(view.buffer().index())
            .ok_or("missing glTF buffer")?;
        let end = view
            .offset()
            .checked_add(view.length())
            .ok_or("glTF buffer view range overflow")?;
        if end > view.buffer().length() || buffer.0.get(view.offset()..end).is_none() {
            return Err(format!("glTF buffer view {} exceeds its buffer", view.index()).into());
        }
    }
    Ok(())
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
            if source.alpha_mode() != gltf::material::AlphaMode::Opaque {
                return Err(
                    "unsupported material: need opaque base-color-only metallic/roughness".into(),
                );
            }
            let factor = pbr.base_color_factor();
            let mut material = Material {
                emissive_texture: None,
                occlusion_texture: None,
                emissive: Vec3::from_array(source.emissive_factor()),
                occlusion_strength: 1.0,
                normal_texture: None,
                metallic_roughness_texture: None,
                normal_scale: 1.0,
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
            let tangents: Option<Vec<_>> = reader.read_tangents().map(|t| t.collect());
            let uv: Option<Vec<_>> = reader
                .read_tex_coords(0)
                .map(|coords| coords.into_f32().collect());
            if tangents
                .as_ref()
                .is_some_and(|t| t.len() != positions.len())
                || normals.len() != positions.len()
                || uv.as_ref().is_some_and(|uv| uv.len() != positions.len())
            {
                return Err("attribute counts differ".into());
            }
            if let Some(tangents) = &tangents
                && tangents.iter().any(|t| {
                    !t.iter().all(|v| v.is_finite())
                        || t[3].abs() != 1.0
                        || Vec3::new(t[0], t[1], t[2]).length_squared() < 1e-8
                })
            {
                return Err("invalid authored tangent".into());
            }
            let mut sampling = govno_render::TextureSampling::default();
            let texture = if let Some(info) = pbr.base_color_texture() {
                if info.tex_coord() != 0 || uv.is_none() {
                    return Err("base-color texture requires TEXCOORD_0".into());
                }
                let sampler = info.texture().sampler();
                sampling = decode_sampler(sampler);
                let data = decode_image(
                    &images[info.texture().source().index()],
                    govno_render::TextureColorSpace::Srgb,
                )?;
                Some(data)
            } else {
                None
            };
            let normal_texture = if let Some(info) = source.normal_texture() {
                if info.tex_coord() != 0 || uv.is_none() {
                    return Err("normal texture requires UV0".into());
                }
                material.normal_scale = info.scale();
                Some((
                    decode_image(
                        &images[info.texture().source().index()],
                        govno_render::TextureColorSpace::Linear,
                    )?,
                    decode_sampler(info.texture().sampler()),
                ))
            } else {
                None
            };
            let metallic_roughness_texture = if let Some(info) = pbr.metallic_roughness_texture() {
                if info.tex_coord() != 0 || uv.is_none() {
                    return Err("metallic/roughness texture requires UV0".into());
                }
                Some((
                    decode_image(
                        &images[info.texture().source().index()],
                        govno_render::TextureColorSpace::Linear,
                    )?,
                    decode_sampler(info.texture().sampler()),
                ))
            } else {
                None
            };
            material.validate()?;
            let emissive_texture = if let Some(info) = source.emissive_texture() {
                if info.tex_coord() != 0 || uv.is_none() {
                    return Err("emissive texture requires UV0".into());
                }
                Some((
                    decode_image(
                        &images[info.texture().source().index()],
                        govno_render::TextureColorSpace::Srgb,
                    )?,
                    decode_sampler(info.texture().sampler()),
                ))
            } else {
                None
            };
            let occlusion_texture = if let Some(info) = source.occlusion_texture() {
                if info.tex_coord() != 0 || uv.is_none() {
                    return Err("occlusion texture requires UV0".into());
                }
                material.occlusion_strength = info.strength();
                Some((
                    decode_image(
                        &images[info.texture().source().index()],
                        govno_render::TextureColorSpace::Linear,
                    )?,
                    decode_sampler(info.texture().sampler()),
                ))
            } else {
                None
            };
            material.validate()?;
            let vertices = positions
                .iter()
                .zip(&normals)
                .enumerate()
                .map(|(i, (position, normal))| Vertex {
                    tangent: tangents.as_ref().map_or([0.0; 4], |t| {
                        let a = t[i];
                        let n = normals_matrix
                            .transform_vector3(Vec3::from_array(*normal))
                            .normalize_or_zero();
                        let t = matrix.transform_vector3(Vec3::new(a[0], a[1], a[2]));
                        let t = (t - n * n.dot(t)).normalize_or_zero();
                        [t.x, t.y, t.z, a[3] * matrix.determinant().signum()]
                    }),
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
                emissive_texture,
                occlusion_texture,
                sampling,
                normal_texture,
                metallic_roughness_texture,
            });
        }
    }
    for child in node.children() {
        visit(child, matrix, depth + 1, buffers, images, output)?;
    }
    Ok(())
}
fn decode_image(
    image: &gltf::image::Data,
    color_space: govno_render::TextureColorSpace,
) -> Result<TextureData, BoxError> {
    let rgba8 = match image.format {
        gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
        gltf::image::Format::R8G8B8 => image
            .pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        _ => return Err("only RGB8/RGBA8 images supported".into()),
    };
    let data = TextureData {
        width: image.width,
        height: image.height,
        rgba8,
        color_space,
    };
    data.validate()?;
    Ok(data)
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
    #[test]
    fn binary_glb_matches_gltf_mesh_material_and_texture() {
        let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/");
        let text = load_gltf(format!("{directory}triangle.gltf")).unwrap();
        let binary = load_gltf(format!("{directory}triangle.glb")).unwrap();
        assert_eq!(binary.len(), text.len());
        assert_eq!(binary[0].mesh.indices, text[0].mesh.indices);
        assert_eq!(binary[0].mesh.vertices.len(), text[0].mesh.vertices.len());
        for (actual, expected) in binary[0].mesh.vertices.iter().zip(&text[0].mesh.vertices) {
            assert_eq!(actual.position, expected.position);
            assert_eq!(actual.normal, expected.normal);
        }
        assert_eq!(binary[0].material.base_color, text[0].material.base_color);
        assert_eq!(
            binary[0].base_color_texture.as_ref().unwrap().rgba8,
            text[0].base_color_texture.as_ref().unwrap().rgba8
        );
    }
    #[test]
    fn image_buffer_view_out_of_bounds_returns_error() {
        let error = load_gltf(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/out-of-bounds-image.gltf"
        ))
        .err()
        .expect("invalid image range must fail");
        assert!(
            error
                .to_string()
                .contains("buffer view 0 exceeds its buffer")
        );
    }
    #[test]
    fn buffer_view_addition_overflow_returns_error() {
        let source = format!(
            r#"{{"asset":{{"version":"2.0"}},"buffers":[{{"byteLength":4}}],"bufferViews":[{{"buffer":0,"byteOffset":{},"byteLength":8}}]}}"#,
            usize::MAX - 3
        );
        let document = gltf::Gltf::from_slice(source.as_bytes()).unwrap().document;
        let buffers = vec![gltf::buffer::Data(vec![0; 4])];
        assert!(validate_buffer_views(&document, &buffers).is_err());
    }
    fn modified(source: &str) -> Result<Vec<ImportedPrimitive>, BoxError> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/triangle.gltf");
        let (_, buffers, images) = gltf::import(path)?;
        let document = gltf::Gltf::from_slice(source.as_bytes())?.document;
        decode(document, buffers, images)
    }
    #[test]
    fn emissive_and_occlusion_maps_preserve_encoding_and_factors() {
        let item = load_gltf(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/emission-occlusion.gltf"
        ))
        .unwrap()
        .remove(0);
        assert!(
            item.material
                .emissive
                .abs_diff_eq(Vec3::new(0.8, 0.3, 0.05), 1e-6)
        );
        assert_eq!(item.material.occlusion_strength, 0.7);
        assert_eq!(
            item.emissive_texture.unwrap().0.color_space,
            govno_render::TextureColorSpace::Srgb
        );
        assert_eq!(
            item.occlusion_texture.unwrap().0.color_space,
            govno_render::TextureColorSpace::Linear
        );
    }
    #[test]
    fn authored_tangents_transform_with_mirrored_nonuniform_scale() {
        let item = load_gltf(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tangents.gltf"
        ))
        .unwrap()
        .remove(0);
        let vertex = &item.mesh.vertices[0];
        assert_eq!(vertex.position, [2.0, 0.0, 0.0]);
        assert_eq!(vertex.tangent, [-1.0, 0.0, 0.0, -1.0]);
        assert_eq!(item.mesh.indices, [0, 2, 1]);
        let fallback = load_gltf(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/triangle.gltf"
        ))
        .unwrap();
        assert_eq!(fallback[0].mesh.vertices[0].tangent, [0.0; 4]);
    }
    #[test]
    fn material_data_maps_are_linear_and_normal_scale_is_preserved() {
        let source = include_str!("../tests/fixtures/triangle.gltf")
            .replace(
                "\"pbrMetallicRoughness\":",
                "\"normalTexture\":{\"index\":0,\"scale\":0.3},\"pbrMetallicRoughness\":",
            )
            .replace(
                "\"metallicFactor\":0",
                "\"metallicRoughnessTexture\":{\"index\":0},\"metallicFactor\":0",
            );
        let item = modified(&source).unwrap().remove(0);
        assert_eq!(item.material.normal_scale, 0.3);
        assert_eq!(
            item.normal_texture.unwrap().0.color_space,
            govno_render::TextureColorSpace::Linear
        );
        assert_eq!(
            item.metallic_roughness_texture.unwrap().0.color_space,
            govno_render::TextureColorSpace::Linear
        );
        assert_eq!(
            item.base_color_texture.unwrap().color_space,
            govno_render::TextureColorSpace::Srgb
        );
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

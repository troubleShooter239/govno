//! CPU-only static glTF/GLB and Radiance HDR decoding. No Vulkan or window dependencies.
mod manager;
pub use manager::{AssetHandle, AssetManager};
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

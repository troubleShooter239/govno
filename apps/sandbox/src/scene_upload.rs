//! Concrete Vulkan upload bridge for authored scenes. CPU scene data stays backend-free.
use govno_core::BoxError;
use govno_render::{DrawInstance, Material, MeshHandle, RenderWorld, TextureHandle};
use govno_render_vulkan::VulkanRenderer;
use govno_scene::PreparedScene;
use std::collections::{HashMap, hash_map::Entry};

#[derive(Default)]
pub struct SceneUpload {
    meshes: Vec<MeshHandle>,
    textures: Vec<TextureHandle>,
}

impl SceneUpload {
    /// Upload each unique CPU model once, then place it any number of times.
    /// Partial uploads are rolled back; the currently displayed scene is untouched.
    pub fn upload(
        renderer: &mut VulkanRenderer,
        scene: &PreparedScene,
    ) -> Result<(Self, Vec<DrawInstance>), BoxError> {
        let mut resources = Self::default();
        let result = resources.upload_inner(renderer, scene);
        match result {
            Ok(instances) => Ok((resources, instances)),
            Err(error) => {
                if let Err(cleanup) = resources.release(renderer) {
                    tracing::error!(%cleanup, "scene upload rollback failed");
                }
                Err(error)
            }
        }
    }

    fn upload_inner(
        &mut self,
        renderer: &mut VulkanRenderer,
        scene: &PreparedScene,
    ) -> Result<Vec<DrawInstance>, BoxError> {
        let mut models = HashMap::new();
        let mut instances = Vec::new();
        if scene.models().is_empty() {
            return Ok(instances);
        }
        let white = renderer.upload_texture(&govno_render::TextureData {
            color_space: Default::default(),
            width: 1,
            height: 1,
            rgba8: vec![255; 4],
        })?;
        self.textures.push(white);
        for model in scene.models() {
            let primitives = match models.entry(model.asset()) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let mut uploaded = Vec::<(MeshHandle, Material)>::new();
                    for primitive in model.data() {
                        let mesh = renderer.upload_mesh(&primitive.mesh)?;
                        self.meshes.push(mesh);
                        let mut material = primitive.material;
                        material.base_color_texture = Some(match &primitive.base_color_texture {
                            Some(data) => self.texture(renderer, data, primitive.sampling)?,
                            None => white,
                        });
                        for (source, destination) in [
                            (&primitive.normal_texture, &mut material.normal_texture),
                            (
                                &primitive.metallic_roughness_texture,
                                &mut material.metallic_roughness_texture,
                            ),
                            (&primitive.emissive_texture, &mut material.emissive_texture),
                            (
                                &primitive.occlusion_texture,
                                &mut material.occlusion_texture,
                            ),
                        ] {
                            if let Some((data, sampling)) = source {
                                *destination = Some(self.texture(renderer, data, *sampling)?);
                            }
                        }
                        uploaded.push((mesh, material));
                    }
                    entry.insert(uploaded)
                }
            };
            instances.extend(primitives.iter().map(|&(mesh, material)| DrawInstance {
                mesh,
                material,
                transform: model.transform().clone(),
            }));
        }
        tracing::info!(
            placements = scene.models().len(),
            meshes = self.meshes.len(),
            textures = self.textures.len(),
            "authored scene uploaded"
        );
        Ok(instances)
    }

    fn texture(
        &mut self,
        renderer: &mut VulkanRenderer,
        data: &govno_render::TextureData,
        sampling: govno_render::TextureSampling,
    ) -> Result<TextureHandle, BoxError> {
        let texture = renderer.upload_texture_with_sampling(data, sampling)?;
        self.textures.push(texture);
        Ok(texture)
    }

    /// Caller first removes all corresponding draw instances. Mesh retirement
    /// uses the renderer's completion serial; texture removal quiesces GPU users.
    pub fn release(mut self, renderer: &mut VulkanRenderer) -> Result<(), BoxError> {
        let mut failure = None;
        for mesh in self.meshes.drain(..).rev() {
            if let Err(error) = renderer.remove_mesh(mesh) {
                failure.get_or_insert(error);
            }
        }
        for texture in self.textures.drain(..).rev() {
            if let Err(error) = renderer.remove_texture(texture) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

pub fn world(scene: &PreparedScene, instances: Vec<DrawInstance>) -> Result<RenderWorld, BoxError> {
    let description = scene.description();
    Ok(RenderWorld {
        frustum_culling: true,
        post_process: Default::default(),
        directional_shadow: Some(Default::default()),
        diffuse_environment: Some(govno_render::DiffuseEnvironment::constant(
            glam::Vec3::ZERO,
        )?),
        specular_environment: true,
        local_probes: vec![],
        camera: description.camera.to_camera(),
        light_direction: glam::Vec3::from_array(description.sun.direction),
        light_radiance: glam::Vec3::from_array(description.sun.radiance),
        point_lights: description
            .point_lights
            .iter()
            .map(|light| light.to_light())
            .collect(),
        instances,
    })
}

/// Preserve the legacy --gltf demo option through the same CPU/GPU scene path.
pub fn prepare_model(
    path: impl AsRef<std::path::Path>,
    assets: &mut govno_scene::ModelAssets,
) -> Result<PreparedScene, BoxError> {
    let path = path.as_ref().canonicalize()?;
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("model filename must be UTF-8")?;
    let description = govno_scene::SceneDescription {
        version: 1,
        name: "Imported model".into(),
        camera: Default::default(),
        sun: Default::default(),
        point_lights: vec![],
        models: vec![govno_scene::ModelDescription {
            id: "imported".into(),
            asset: filename.into(),
            transform: Default::default(),
        }],
    };
    PreparedScene::prepare(
        description,
        path.parent().ok_or("model path has no parent")?,
        assets,
    )
}

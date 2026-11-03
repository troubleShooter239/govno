use crate::SceneDescription;
use govno_assets::{AssetHandle, AssetManager, ImportedPrimitive};
use govno_core::BoxError;
use govno_render::Transform;
use std::{path::Path, sync::Arc};

pub type ModelAssets = AssetManager<Vec<ImportedPrimitive>>;

/// Keeps CPU model data alive until the prepared scene is dropped.
pub struct PreparedModel {
    id: String,
    asset: AssetHandle<Vec<ImportedPrimitive>>,
    data: Arc<Vec<ImportedPrimitive>>,
    transform: Transform,
}
impl PreparedModel {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn asset(&self) -> AssetHandle<Vec<ImportedPrimitive>> {
        self.asset
    }
    pub fn data(&self) -> &[ImportedPrimitive] {
        &self.data
    }
    pub fn transform(&self) -> &Transform {
        &self.transform
    }
}

/// Immutable CPU preparation result. Does not mutate a running world on failure.
/// Holding multiple scenes shares decoded models through the caller's cache.
/// Drop + AssetManager::remove_unused explicitly unloads unreferenced CPU data.
pub struct PreparedScene {
    description: SceneDescription,
    models: Vec<PreparedModel>,
}
impl PreparedScene {
    pub fn load(path: impl AsRef<Path>, assets: &mut ModelAssets) -> Result<Self, BoxError> {
        let path = path.as_ref().canonicalize()?;
        let description = SceneDescription::load(&path)?;
        let directory = path.parent().ok_or("scene path has no parent directory")?;
        Self::prepare(description, directory, assets)
    }

    pub fn prepare(
        description: SceneDescription,
        directory: &Path,
        assets: &mut ModelAssets,
    ) -> Result<Self, BoxError> {
        description.validate()?;
        let mut models = Vec::with_capacity(description.models.len());
        for model in &description.models {
            let asset = assets
                .load(directory.join(&model.asset), |path| {
                    govno_assets::load_gltf(path)
                })
                .map_err(|error| -> BoxError {
                    format!("scene model {:?} ({}): {error}", model.id, model.asset).into()
                })?;
            let data = assets
                .get(asset)
                .ok_or("model cache lost a newly loaded asset")?
                .clone();
            models.push(PreparedModel {
                id: model.id.clone(),
                asset,
                data,
                transform: model.transform.to_transform(),
            });
        }
        // Successful partial loads remain cacheable on failure. Callers may prune
        // unused cache entries; already-active scenes retain their Arc ownership.
        Ok(Self {
            description,
            models,
        })
    }

    pub fn description(&self) -> &SceneDescription {
        &self.description
    }
    pub fn models(&self) -> &[PreparedModel] {
        &self.models
    }
}

//! Typed CPU asset ownership and explicit cache eviction. No GPU backend dependency.
use govno_core::BoxError;
use std::{
    collections::HashMap,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_MANAGER: AtomicU64 = AtomicU64::new(1);

/// Typed, manager-scoped ID. IDs are never reused, so removed handles stay stale.
pub struct AssetHandle<T> {
    owner: u64,
    id: u64,
    marker: PhantomData<fn() -> T>,
}
impl<T> Copy for AssetHandle<T> {}
impl<T> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> PartialEq for AssetHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.owner == other.owner && self.id == other.id
    }
}
impl<T> Eq for AssetHandle<T> {}
impl<T> Hash for AssetHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.owner.hash(state);
        self.id.hash(state);
    }
}
impl<T> fmt::Debug for AssetHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AssetHandle")
            .field("owner", &self.owner)
            .field("id", &self.id)
            .finish()
    }
}

struct Entry<T> {
    path: PathBuf,
    value: Arc<T>,
}

/// Load-time cache. A successful load is shared through Arc; failed loads are
/// not cached. Callers own uploads, GPU retirement and scene instantiation.
/// Loading is deliberately synchronous in this first stage-03 increment.
pub struct AssetManager<T> {
    owner: u64,
    next: u64,
    paths: HashMap<PathBuf, u64>,
    entries: HashMap<u64, Entry<T>>,
}
impl<T> Default for AssetManager<T> {
    fn default() -> Self {
        Self {
            owner: NEXT_MANAGER
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("asset manager identity space exhausted"),
            next: 0,
            paths: HashMap::new(),
            entries: HashMap::new(),
        }
    }
}
impl<T> AssetManager<T> {
    /// Canonical paths collapse relative aliases/symlinks before cache lookup.
    /// The supplied CPU loader runs only on a cache miss. Changes on disk require
    /// explicit remove + load; hot reload belongs to a later milestone.
    pub fn load(
        &mut self,
        path: impl AsRef<Path>,
        loader: impl FnOnce(&Path) -> Result<T, BoxError>,
    ) -> Result<AssetHandle<T>, BoxError> {
        let path = path.as_ref().canonicalize()?;
        let id = if let Some(&id) = self.paths.get(&path) {
            id
        } else {
            let next = self
                .next
                .checked_add(1)
                .ok_or("asset identity space exhausted")?;
            let value = Arc::new(loader(&path)?);
            let id = self.next;
            self.next = next;
            self.paths.insert(path.clone(), id);
            self.entries.insert(id, Entry { path, value });
            id
        };
        Ok(AssetHandle {
            owner: self.owner,
            id,
            marker: PhantomData,
        })
    }

    pub fn get(&self, handle: AssetHandle<T>) -> Option<&Arc<T>> {
        (handle.owner == self.owner)
            .then(|| self.entries.get(&handle.id))
            .flatten()
            .map(|entry| &entry.value)
    }

    /// Evicts the cached identity. Existing Arc readers retain their data; old
    /// IDs are invalid even if the same file is immediately loaded again.
    pub fn remove(&mut self, handle: AssetHandle<T>) -> Option<Arc<T>> {
        if handle.owner != self.owner {
            return None;
        }
        let entry = self.entries.remove(&handle.id)?;
        self.paths.remove(&entry.path);
        Some(entry.value)
    }

    /// Evicts entries owned only by this cache. Handles are non-owning IDs:
    /// callers that need to retain data must hold an Arc obtained from get().
    /// Safe even if another thread holds an Arc; it cannot acquire a new one
    /// from this manager while this exclusive borrow is active.
    pub fn remove_unused(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|id, entry| {
            if Arc::strong_count(&entry.value) > 1 {
                true
            } else {
                debug_assert_eq!(self.paths.get(&entry.path), Some(id));
                self.paths.remove(&entry.path);
                false
            }
        });
        before - self.entries.len()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

use ash::vk;
use govno_core::BoxError;
use std::{ffi::CStr, path::PathBuf};
#[derive(Debug, Clone, Copy)]
pub enum DeviceProfile {
    Modern,
    Advanced,
    RayTracing,
}
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub modern: bool,
    pub advanced: bool,
    pub ray_tracing: bool,
    pub timeline_semaphore: bool,
    pub descriptor_indexing: bool,
    pub buffer_device_address: bool,
    pub max_push_constants: u32,
    pub max_image_dimension: u32,
    pub timestamp_period: f32,
    pub timestamp_valid_bits: u32,
    pub local_heap_bytes: u64,
    pub memory_budget: bool,
}
impl DeviceCapabilities {
    pub fn supports(&self, profile: DeviceProfile) -> bool {
        match profile {
            DeviceProfile::Modern => self.modern,
            DeviceProfile::Advanced => self.advanced,
            DeviceProfile::RayTracing => self.ray_tracing,
        }
    }
}
pub(crate) fn probe(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    family: u32,
) -> DeviceCapabilities {
    // SAFETY: caller supplies a physical device and queue family from this instance.
    unsafe {
        let properties = instance.get_physical_device_properties(physical);
        let q = instance.get_physical_device_queue_family_properties(physical)[family as usize];
        let extensions = instance
            .enumerate_device_extension_properties(physical)
            .unwrap_or_default();
        let has = |name: &CStr| {
            extensions
                .iter()
                .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == name)
        };
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut acceleration = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut rt = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let rt_extensions = has(c"VK_KHR_acceleration_structure")
            && has(c"VK_KHR_ray_tracing_pipeline")
            && has(c"VK_KHR_deferred_host_operations");
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut v12)
            .push_next(&mut v13);
        if rt_extensions {
            features = features.push_next(&mut acceleration).push_next(&mut rt);
        }
        instance.get_physical_device_features2(physical, &mut features);
        let format = |f: vk::Format, flags: vk::FormatFeatureFlags| {
            instance
                .get_physical_device_format_properties(physical, f)
                .optimal_tiling_features
                .contains(flags)
        };
        let linear = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        let formats = format(
            vk::Format::D32_SFLOAT,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | linear,
        ) && format(
            vk::Format::R16G16B16A16_SFLOAT,
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::STORAGE_IMAGE
                | linear,
        ) && format(
            vk::Format::R32G32B32A32_SFLOAT,
            linear
                | vk::FormatFeatureFlags::STORAGE_IMAGE
                | vk::FormatFeatureFlags::TRANSFER_SRC
                | vk::FormatFeatureFlags::TRANSFER_DST,
        ) && [vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM]
            .iter()
            .all(|&f| {
                format(
                    f,
                    linear | vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST,
                )
            });
        let l = properties.limits;
        let limits = l.max_push_constants_size >= 128
            && l.max_bound_descriptor_sets >= 4
            && l.max_uniform_buffer_range >= 1248
            && l.max_image_dimension2_d >= 4096
            && l.max_per_stage_descriptor_storage_buffers >= 4
            && l.max_per_stage_descriptor_sampled_images >= 20
            && l.max_per_stage_descriptor_samplers >= 12;
        let modern = properties.api_version >= vk::API_VERSION_1_3
            && v12.timeline_semaphore != 0
            && v13.dynamic_rendering != 0
            && v13.synchronization2 != 0
            && formats
            && limits
            && q.timestamp_valid_bits > 0;
        let indexing = v12.runtime_descriptor_array != 0
            && v12.descriptor_binding_partially_bound != 0
            && v12.descriptor_binding_sampled_image_update_after_bind != 0;
        let advanced =
            modern && indexing && v12.buffer_device_address != 0 && v12.draw_indirect_count != 0;
        let memory = instance.get_physical_device_memory_properties(physical);
        let local_heap_bytes = memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();
        DeviceCapabilities {
            name: CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned(),
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            api_version: properties.api_version,
            driver_version: properties.driver_version,
            modern,
            advanced,
            ray_tracing: advanced
                && rt_extensions
                && acceleration.acceleration_structure != 0
                && rt.ray_tracing_pipeline != 0,
            timeline_semaphore: v12.timeline_semaphore != 0,
            descriptor_indexing: indexing,
            buffer_device_address: v12.buffer_device_address != 0,
            max_push_constants: l.max_push_constants_size,
            max_image_dimension: l.max_image_dimension2_d,
            timestamp_period: l.timestamp_period,
            timestamp_valid_bits: q.timestamp_valid_bits,
            local_heap_bytes,
            memory_budget: has(c"VK_EXT_memory_budget"),
        }
    }
}
pub(crate) fn cache_path(p: &vk::PhysicalDeviceProperties) -> PathBuf {
    let uuid = p
        .pipeline_cache_uuid
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    std::env::var_os("GOVNO_PIPELINE_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/pipeline-cache"))
        .join(format!(
            "{:x}-{:x}-{}-{uuid}.bin",
            p.vendor_id, p.device_id, p.driver_version
        ))
}
pub(crate) fn cache_compatible(data: &[u8], p: &vk::PhysicalDeviceProperties) -> bool {
    data.len() >= 32
        && u32::from_le_bytes(data[0..4].try_into().unwrap()) == 32
        && u32::from_le_bytes(data[4..8].try_into().unwrap()) == 1
        && u32::from_le_bytes(data[8..12].try_into().unwrap()) == p.vendor_id
        && u32::from_le_bytes(data[12..16].try_into().unwrap()) == p.device_id
        && data[16..32] == p.pipeline_cache_uuid
}
pub(crate) fn create_cache(
    device: &ash::Device,
    p: &vk::PhysicalDeviceProperties,
) -> Result<(vk::PipelineCache, PathBuf, u64), BoxError> {
    let path = cache_path(p);
    let bytes = std::fs::read(&path)
        .ok()
        .filter(|bytes| cache_compatible(bytes, p))
        .unwrap_or_default();
    // SAFETY: host cache access is serialized, header compatibility checked; driver validates opaque data.
    let (cache, loaded) = unsafe {
        match device.create_pipeline_cache(
            &vk::PipelineCacheCreateInfo::default().initial_data(&bytes),
            None,
        ) {
            Ok(c) => (c, bytes.len() as u64),
            Err(e) if !bytes.is_empty() => {
                tracing::warn!(?e, "pipeline cache rejected; using empty cache");
                (
                    device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?,
                    0,
                )
            }
            Err(e) => return Err(e.into()),
        }
    };
    tracing::info!(bytes=loaded,path=%path.display(),"pipeline cache loaded");
    Ok((cache, path, loaded))
}
pub(crate) fn save_cache(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(temporary, path)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_header_rejects_other_devices_and_truncation() {
        let mut p = vk::PhysicalDeviceProperties {
            vendor_id: 123,
            device_id: 456,
            ..Default::default()
        };
        p.pipeline_cache_uuid = [7; 16];
        let mut data: Vec<u8> = [32_u32, 1, 123, 456]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        data.extend([7; 16]);
        assert!(cache_compatible(&data, &p));
        data.extend([1, 2, 3, 4]);
        assert!(cache_compatible(&data, &p));
        assert!(!cache_compatible(&data[..16], &p));
        data[8] = 0;
        assert!(!cache_compatible(&data, &p));
    }
}

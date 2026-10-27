use ash::{Entry, vk};
use govno_core::BoxError;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::{
    ffi::{CStr, c_void},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

pub trait SurfaceTarget: HasDisplayHandle + HasWindowHandle + Send + Sync {}
impl<T: HasDisplayHandle + HasWindowHandle + Send + Sync> SurfaceTarget for T {}

#[derive(Default)]
pub struct ValidationStats {
    errors: AtomicUsize,
    warnings: AtomicUsize,
}
impl ValidationStats {
    pub fn errors(&self) -> usize {
        self.errors.load(Ordering::Relaxed)
    }
    pub fn warnings(&self) -> usize {
        self.warnings.load(Ordering::Relaxed)
    }
}
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: Vulkan supplies valid callback data; user points to the Arc-owned
    // ValidationStats retained until after the messenger and instance are destroyed.
    unsafe {
        let _ = std::panic::catch_unwind(|| {
            if data.is_null() || user.is_null() {
                return;
            }
            let stats = &*user.cast::<ValidationStats>();
            if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
                stats.errors.fetch_add(1, Ordering::Relaxed);
            } else {
                stats.warnings.fetch_add(1, Ordering::Relaxed);
            }
            if !(*data).p_message.is_null() {
                let message = CStr::from_ptr((*data).p_message).to_string_lossy();
                if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
                    tracing::error!(%message, "Vulkan validation");
                } else {
                    tracing::warn!(%message, "Vulkan validation");
                }
            }
        });
    }
    vk::FALSE
}
struct InstanceContext {
    _entry: Entry,
    instance: ash::Instance,
    debug: Option<ash::ext::debug_utils::Instance>,
    messenger: vk::DebugUtilsMessengerEXT,
    _stats: Arc<ValidationStats>,
}
impl Drop for InstanceContext {
    fn drop(&mut self) {
        // SAFETY: all surfaces/devices retain this owner and have been destroyed.
        unsafe {
            if let Some(debug) = &self.debug {
                debug.destroy_debug_utils_messenger(self.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
struct SurfaceContext {
    instance: Arc<InstanceContext>,
    loader: ash::khr::surface::Instance,
    handle: vk::SurfaceKHR,
    _window: Arc<dyn SurfaceTarget>,
}
impl Drop for SurfaceContext {
    fn drop(&mut self) {
        // SAFETY: the window/instance are retained, all dependent devices are gone.
        unsafe {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}
pub(crate) struct DeviceContext {
    surface: Arc<SurfaceContext>,
    pub device: ash::Device,
    pub swapchain: ash::khr::swapchain::Device,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub family: u32,
    pub present_fences: bool,
}
impl DeviceContext {
    pub fn instance(&self) -> &ash::Instance {
        &self.surface.instance.instance
    }
    pub fn depth_format(&self) -> Result<vk::Format, BoxError> {
        // SAFETY: the physical device belongs to this retained instance.
        let properties = unsafe {
            self.instance()
                .get_physical_device_format_properties(self.physical, vk::Format::D32_SFLOAT)
        };
        if properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            Ok(vk::Format::D32_SFLOAT)
        } else {
            Err("D32_SFLOAT depth attachment required for reversed-Z".into())
        }
    }
    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface.handle
    }
    pub fn capabilities(&self) -> Result<vk::SurfaceCapabilitiesKHR, BoxError> {
        // SAFETY: physical device and surface belong to the retained instance.
        Ok(unsafe {
            self.surface
                .loader
                .get_physical_device_surface_capabilities(self.physical, self.surface.handle)?
        })
    }
    pub fn formats(&self) -> Result<Vec<vk::SurfaceFormatKHR>, BoxError> {
        // SAFETY: physical device and surface belong to the retained instance.
        Ok(unsafe {
            self.surface
                .loader
                .get_physical_device_surface_formats(self.physical, self.surface.handle)?
        })
    }
    pub fn wait_idle(&self) -> Result<(), BoxError> {
        // SAFETY: renderer serializes all queue access on its owning thread.
        unsafe { self.device.device_wait_idle()? };
        Ok(())
    }
}
impl Drop for DeviceContext {
    fn drop(&mut self) {
        // SAFETY: all child owners are gone; no further submission can occur.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

pub(crate) fn create_device(
    window: Arc<dyn SurfaceTarget>,
    validation: bool,
    stats: Arc<ValidationStats>,
) -> Result<Arc<DeviceContext>, BoxError> {
    // SAFETY: loader lifetime is retained by InstanceContext. All Vulkan handles
    // created below are immediately transferred to RAII owners on success.
    unsafe {
        let entry = Entry::load()?;
        let loader_version = entry
            .try_enumerate_instance_version()?
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < vk::API_VERSION_1_3 {
            return Err("Vulkan 1.3 loader required".into());
        }
        let version = loader_version.min(vk::make_api_version(0, 1, 4, 0));
        let display = window.display_handle()?.as_raw();
        let mut extensions = ash_window::enumerate_required_extensions(display)?.to_vec();
        let supported = entry.enumerate_instance_extension_properties(None)?;
        let has_extension = |name: &CStr| {
            supported
                .iter()
                .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == name)
        };
        let maintenance_surface = has_extension(c"VK_EXT_surface_maintenance1")
            && has_extension(c"VK_KHR_get_surface_capabilities2");
        if maintenance_surface {
            extensions.push(c"VK_EXT_surface_maintenance1".as_ptr());
            extensions.push(c"VK_KHR_get_surface_capabilities2".as_ptr());
        }
        let layers = if validation {
            let available = entry.enumerate_instance_layer_properties()?;
            if !available
                .iter()
                .any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == c"VK_LAYER_KHRONOS_validation")
            {
                return Err(
                    "validation requested but VK_LAYER_KHRONOS_validation is unavailable".into(),
                );
            }
            if !has_extension(ash::ext::debug_utils::NAME) {
                return Err("VK_EXT_debug_utils unavailable".into());
            }
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
        } else {
            Vec::new()
        };
        let app = vk::ApplicationInfo::default()
            .application_name(c"GOVNO Sandbox")
            .engine_name(c"GOVNO")
            .api_version(version);
        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
            .user_data(Arc::as_ptr(&stats).cast_mut().cast());
        let mut info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        if validation {
            info = info.push_next(&mut debug_info);
        }
        let instance = entry.create_instance(&info, None)?;
        let mut owner = InstanceContext {
            _entry: entry,
            instance,
            debug: None,
            messenger: vk::DebugUtilsMessengerEXT::null(),
            _stats: stats,
        };
        if validation {
            let debug = ash::ext::debug_utils::Instance::new(&owner._entry, &owner.instance);
            owner.messenger = debug.create_debug_utils_messenger(&debug_info, None)?;
            owner.debug = Some(debug);
        }
        let instance = Arc::new(owner);
        let loader = ash::khr::surface::Instance::new(&instance._entry, &instance.instance);
        let handle = ash_window::create_surface(
            &instance._entry,
            &instance.instance,
            display,
            window.window_handle()?.as_raw(),
            None,
        )?;
        let surface = Arc::new(SurfaceContext {
            instance,
            loader,
            handle,
            _window: window,
        });
        let instance = &surface.instance.instance;
        let mut devices = instance.enumerate_physical_devices()?;
        devices.sort_by_key(|device| {
            let p = instance.get_physical_device_properties(*device);
            if p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                0
            } else {
                1
            }
        });
        for physical in devices {
            let properties = instance.get_physical_device_properties(physical);
            let name = CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy();
            if properties.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let available = instance.enumerate_device_extension_properties(physical)?;
            let has = |name: &CStr| {
                available
                    .iter()
                    .any(|p| CStr::from_ptr(p.extension_name.as_ptr()) == name)
            };
            if !has(ash::khr::swapchain::NAME) {
                continue;
            }
            let maintenance_available =
                maintenance_surface && has(ash::ext::swapchain_maintenance1::NAME);
            let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut maintenance = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut v13);
            if maintenance_available {
                features = features.push_next(&mut maintenance);
            }
            instance.get_physical_device_features2(physical, &mut features);
            if v13.dynamic_rendering == vk::FALSE || v13.synchronization2 == vk::FALSE {
                continue;
            }
            let mut family = None;
            for (index, properties) in instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .enumerate()
            {
                if properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    && surface.loader.get_physical_device_surface_support(
                        physical,
                        index as u32,
                        surface.handle,
                    )?
                {
                    family = Some(index as u32);
                    break;
                }
            }
            let Some(family) = family else {
                continue;
            };
            if surface
                .loader
                .get_physical_device_surface_formats(physical, surface.handle)?
                .is_empty()
                || surface
                    .loader
                    .get_physical_device_surface_present_modes(physical, surface.handle)?
                    .is_empty()
            {
                continue;
            }
            let caps = surface
                .loader
                .get_physical_device_surface_capabilities(physical, surface.handle)?;
            if !caps
                .supported_usage_flags
                .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            {
                continue;
            }
            let priorities = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities)];
            let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];
            let present_fences =
                maintenance_available && maintenance.swapchain_maintenance1 == vk::TRUE;
            let mut enabled_maintenance =
                vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                    .swapchain_maintenance1(true);
            let mut enabled_v13 = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            if present_fences {
                device_extensions.push(ash::ext::swapchain_maintenance1::NAME.as_ptr());
            }
            let mut device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions)
                .push_next(&mut enabled_v13);
            if present_fences {
                device_info = device_info.push_next(&mut enabled_maintenance);
            }
            let device = instance.create_device(physical, &device_info, None)?;
            let queue = device.get_device_queue(family, 0);
            let swapchain = ash::khr::swapchain::Device::new(instance, &device);
            tracing::info!(gpu = %name, api_major = vk::api_version_major(properties.api_version),
                api_minor = vk::api_version_minor(properties.api_version), present_fences, validation, "Vulkan device selected");
            if !present_fences {
                tracing::warn!(
                    "No swapchain maintenance1: resize/shutdown use conventional device-idle WSI fallback"
                );
            }
            return Ok(Arc::new(DeviceContext {
                surface,
                device,
                physical,
                queue,
                family,
                swapchain,
                present_fences,
            }));
        }
        Err("no compatible GPU: need Vulkan 1.3, dynamicRendering, synchronization2, swapchain and a graphics+present queue".into())
    }
}

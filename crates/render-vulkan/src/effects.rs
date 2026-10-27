use crate::{
    compute::Compute,
    context::DeviceContext,
    memory::{Buffer, ImageResource, Memory},
    swapchain::color_range,
    sync,
};
use ash::vk;
use govno_core::BoxError;
use govno_render::PostProcessSettings;
use gpu_allocator::MemoryLocation;
use std::{rc::Rc, sync::Arc};
pub(crate) struct Effects {
    ctx: Arc<DeviceContext>,
    pub bloom: Vec<ImageResource>,
    scratch: Vec<ImageResource>,
    pub exposure: Buffer,
    meter: Compute,
    extract: Compute,
    blur: Compute,
    extent: vk::Extent2D,
}
impl Effects {
    pub fn new(
        memory: Rc<Memory>,
        extent: vk::Extent2D,
        hdr: &[ImageResource],
    ) -> Result<Self, BoxError> {
        let ctx = memory.ctx.clone();
        // SAFETY: physical device is retained by ctx.
        let features = unsafe {
            ctx.instance().get_physical_device_format_properties(
                ctx.physical,
                vk::Format::R16G16B16A16_SFLOAT,
            )
        }
        .optimal_tiling_features;
        if !features
            .contains(vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::SAMPLED_IMAGE)
        {
            return Err("RGBA16F storage/sampled images required for bloom".into());
        }
        let size = vk::Extent2D {
            width: extent.width.div_ceil(2),
            height: extent.height.div_ceil(2),
        };
        let bindings = [
            (0, vk::DescriptorType::SAMPLED_IMAGE),
            (1, vk::DescriptorType::STORAGE_BUFFER),
            (2, vk::DescriptorType::STORAGE_IMAGE),
        ];
        let mut owner = Self {
            ctx: ctx.clone(),
            bloom: vec![],
            scratch: vec![],
            exposure: Buffer::new(
                memory.clone(),
                16,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?,
            meter: Compute::new(
                ctx.clone(),
                include_bytes!("../shaders/meterExposure.spv"),
                c"meterExposure",
                &bindings,
                48,
                2,
            )?,
            extract: Compute::new(
                ctx.clone(),
                include_bytes!("../shaders/bloomExtract.spv"),
                c"bloomExtract",
                &bindings,
                48,
                2,
            )?,
            blur: Compute::new(
                ctx.clone(),
                include_bytes!("../shaders/bloomBlur.spv"),
                c"bloomBlur",
                &bindings,
                48,
                4,
            )?,
            extent: size,
        };
        owner.exposure.write(
            &[1.0_f32, 1.0, 0.0, 0.0]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        for _ in 0..2 {
            let create = || {
                ImageResource::with_usage(
                    memory.clone(),
                    size,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                    color_range(),
                )
            };
            owner.bloom.push(create()?);
            owner.scratch.push(create()?);
        }
        for (i, source) in hdr.iter().enumerate() {
            owner.bind(
                owner.meter.sets[i],
                source,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &owner.bloom[i],
            );
            owner.bind(
                owner.extract.sets[i],
                source,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &owner.bloom[i],
            );
            owner.bind(
                owner.blur.sets[i * 2],
                &owner.bloom[i],
                vk::ImageLayout::GENERAL,
                &owner.scratch[i],
            );
            owner.bind(
                owner.blur.sets[i * 2 + 1],
                &owner.scratch[i],
                vk::ImageLayout::GENERAL,
                &owner.bloom[i],
            );
        }
        Ok(owner)
    }
    fn bind(
        &self,
        set: vk::DescriptorSet,
        input: &ImageResource,
        layout: vk::ImageLayout,
        output: &ImageResource,
    ) {
        let input = [vk::DescriptorImageInfo::default()
            .image_view(input.view)
            .image_layout(layout)];
        let output = [vk::DescriptorImageInfo::default()
            .image_view(output.view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let state = [vk::DescriptorBufferInfo::default()
            .buffer(self.exposure.handle)
            .range(16)];
        // SAFETY: new immutable descriptors retain all referenced resources until quiesced drop.
        unsafe {
            self.ctx.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&input),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&state),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&output),
                ],
                &[],
            );
        }
    }
    pub fn record(&self, cmd: vk::CommandBuffer, frame: usize, settings: PostProcessSettings) {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        let a = settings.auto_exposure.unwrap_or_default();
        let mut params = [
            settings.exposure_multiplier(),
            a.key,
            a.min_ev.exp2(),
            a.max_ev.exp2(),
            settings.bloom.threshold,
            settings.bloom.knee,
            settings.bloom.intensity,
            if settings.auto_exposure.is_some() {
                1.0
            } else {
                0.0
            },
            settings.delta_seconds,
            a.speed,
            0.0,
            0.0,
        ];
        let bytes = |p: &[f32]| p.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>();
        sync::memory(
            &self.ctx,
            cmd,
            (
                S::COMPUTE_SHADER | S::FRAGMENT_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
            ),
            (S::COMPUTE_SHADER, A::SHADER_READ | A::SHADER_WRITE),
        );
        self.meter.dispatch(cmd, frame, &bytes(&params), [1, 1, 1]);
        sync::memory(
            &self.ctx,
            cmd,
            (S::COMPUTE_SHADER, A::SHADER_WRITE),
            (S::COMPUTE_SHADER | S::FRAGMENT_SHADER, A::SHADER_READ),
        );
        for image in [&self.bloom[frame], &self.scratch[frame]] {
            sync::transition(
                &self.ctx,
                cmd,
                image.image,
                color_range(),
                (
                    S::COMPUTE_SHADER | S::FRAGMENT_SHADER,
                    A::SHADER_READ | A::SHADER_WRITE,
                    L::UNDEFINED,
                ),
                (
                    S::COMPUTE_SHADER,
                    A::SHADER_READ | A::SHADER_WRITE,
                    L::GENERAL,
                ),
            );
        }
        let groups = [
            self.extent.width.div_ceil(8),
            self.extent.height.div_ceil(8),
            1,
        ];
        self.extract.dispatch(cmd, frame, &bytes(&params), groups);
        sync::memory(
            &self.ctx,
            cmd,
            (S::COMPUTE_SHADER, A::SHADER_WRITE),
            (S::COMPUTE_SHADER, A::SHADER_READ | A::SHADER_WRITE),
        );
        self.blur.dispatch(cmd, frame * 2, &bytes(&params), groups);
        sync::memory(
            &self.ctx,
            cmd,
            (S::COMPUTE_SHADER, A::SHADER_READ | A::SHADER_WRITE),
            (S::COMPUTE_SHADER, A::SHADER_READ | A::SHADER_WRITE),
        );
        params[11] = 1.0;
        self.blur
            .dispatch(cmd, frame * 2 + 1, &bytes(&params), groups);
        sync::transition(
            &self.ctx,
            cmd,
            self.bloom[frame].image,
            color_range(),
            (S::COMPUTE_SHADER, A::SHADER_WRITE, L::GENERAL),
            (
                S::FRAGMENT_SHADER,
                A::SHADER_READ,
                L::SHADER_READ_ONLY_OPTIMAL,
            ),
        );
    }
}

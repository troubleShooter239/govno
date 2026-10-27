use crate::{
    compute::Compute,
    memory::{Buffer, ImageResource, Memory},
    mesh::Upload,
    swapchain::color_range,
    sync,
};
use ash::vk;
use govno_core::BoxError;
use govno_render::{
    BRDF_LUT_SIZE, EnvironmentRadiance, SPECULAR_HEIGHT, SPECULAR_LEVELS, SPECULAR_WIDTH,
    SpecularEnvironment,
};
use gpu_allocator::MemoryLocation;
use std::rc::Rc;
pub(crate) const PROBE_SLOTS: u32 = 3;
pub(crate) struct SpecularGpu {
    memory: Rc<Memory>,
    pub radiance: ImageResource,
    lut: ImageResource,
    sampler: vk::Sampler,
    pub sh: Buffer,
    prefilter: Compute,
    brdf: Compute,
    project: Compute,
    initialized: bool,
}
impl SpecularGpu {
    pub fn new(memory: Rc<Memory>) -> Result<Self, BoxError> {
        let ctx = memory.ctx.clone();
        // SAFETY: physical device belongs to retained ctx.
        let features = unsafe {
            ctx.instance().get_physical_device_format_properties(
                ctx.physical,
                vk::Format::R32G32B32A32_SFLOAT,
            )
        }
        .optimal_tiling_features;
        if !features.contains(
            vk::FormatFeatureFlags::STORAGE_IMAGE
                | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
        ) {
            return Err("RGBA32F storage/filter support required for GPU IBL".into());
        }
        let create = |width, height| {
            ImageResource::with_usage(
                memory.clone(),
                vk::Extent2D { width, height },
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                color_range(),
            )
        };
        let bindings = [
            (0, vk::DescriptorType::SAMPLED_IMAGE),
            (1, vk::DescriptorType::SAMPLER),
            (2, vk::DescriptorType::STORAGE_IMAGE),
            (3, vk::DescriptorType::STORAGE_IMAGE),
            (4, vk::DescriptorType::STORAGE_BUFFER),
        ];
        let mut owner = Self {
            memory: memory.clone(),
            radiance: create(
                SPECULAR_WIDTH,
                SPECULAR_HEIGHT * SPECULAR_LEVELS * PROBE_SLOTS,
            )?,
            lut: create(BRDF_LUT_SIZE, BRDF_LUT_SIZE)?,
            sampler: vk::Sampler::null(),
            sh: Buffer::new(
                memory.clone(),
                PROBE_SLOTS as u64 * 9 * 16,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
            )?,
            prefilter: Compute::new(
                ctx.clone(),
                include_bytes!("../shaders/prefilterIBL.spv"),
                c"prefilterIBL",
                &bindings,
                16,
                1,
            )?,
            brdf: Compute::new(
                ctx.clone(),
                include_bytes!("../shaders/integrateBRDF.spv"),
                c"integrateBRDF",
                &bindings,
                16,
                1,
            )?,
            project: Compute::new(
                ctx.clone(),
                include_bytes!("../shaders/projectSH.spv"),
                c"projectSH",
                &bindings,
                16,
                1,
            )?,
            initialized: false,
        };
        // SAFETY: sampler has immediate RAII ownership and uses a single initialized LOD.
        owner.sampler = unsafe {
            ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .max_lod(0.0)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let black = EnvironmentRadiance::constant(glam::Vec3::ZERO)?;
        for i in 0..PROBE_SLOTS {
            owner.process(&black, i)?;
        }
        Ok(owner)
    }
    pub fn process(&mut self, data: &EnvironmentRadiance, probe: u32) -> Result<(), BoxError> {
        if probe >= PROBE_SLOTS {
            return Err("local probe index exceeds two supported probes".into());
        }
        let (width, height) = data.dimensions();
        let pixels: Vec<_> = data
            .pixels()
            .iter()
            .map(|p| p.extend(1.0).to_array())
            .collect();
        let source = upload(self.memory.clone(), width, height, &pixels)?;
        let ctx = self.memory.ctx.clone();
        for pipeline in [&self.prefilter, &self.brdf, &self.project] {
            let input = [vk::DescriptorImageInfo::default()
                .image_view(source.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let sampler = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
            let atlas = [vk::DescriptorImageInfo::default()
                .image_view(self.radiance.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let lut = [vk::DescriptorImageInfo::default()
                .image_view(self.lut.view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let sh = [vk::DescriptorBufferInfo::default()
                .buffer(self.sh.handle)
                .range(PROBE_SLOTS as u64 * 9 * 16)];
            // SAFETY: synchronous upload entry point quiesces prior users before descriptor mutation.
            unsafe {
                ctx.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(pipeline.sets[0])
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&input),
                        vk::WriteDescriptorSet::default()
                            .dst_set(pipeline.sets[0])
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler),
                        vk::WriteDescriptorSet::default()
                            .dst_set(pipeline.sets[0])
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&atlas),
                        vk::WriteDescriptorSet::default()
                            .dst_set(pipeline.sets[0])
                            .dst_binding(3)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&lut),
                        vk::WriteDescriptorSet::default()
                            .dst_set(pipeline.sets[0])
                            .dst_binding(4)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&sh),
                    ],
                    &[],
                );
            }
        }
        let push: Vec<u8> = [probe, 128, 0, 0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        Upload::run(ctx.clone(), |cmd| {
            crate::sync::memory(
                &ctx,
                cmd,
                (
                    vk::PipelineStageFlags2::COPY
                        | vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::TRANSFER_WRITE
                        | vk::AccessFlags2::SHADER_READ
                        | vk::AccessFlags2::SHADER_WRITE,
                ),
                (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                ),
            );
            use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
            sync::transition(
                &ctx,
                cmd,
                self.radiance.image,
                color_range(),
                (
                    S::FRAGMENT_SHADER,
                    A::SHADER_READ,
                    if self.initialized {
                        L::SHADER_READ_ONLY_OPTIMAL
                    } else {
                        L::UNDEFINED
                    },
                ),
                (S::COMPUTE_SHADER, A::SHADER_WRITE, L::GENERAL),
            );
            if !self.initialized {
                sync::transition(
                    &ctx,
                    cmd,
                    self.lut.image,
                    color_range(),
                    (S::NONE, A::NONE, L::UNDEFINED),
                    (S::COMPUTE_SHADER, A::SHADER_WRITE, L::GENERAL),
                );
                self.brdf.dispatch(cmd, 0, &push, [8, 8, 1]);
            }
            self.prefilter.dispatch(cmd, 0, &push, [8, 4, 6]);
            self.project.dispatch(cmd, 0, &push, [1, 1, 1]);
            sync::transition(
                &ctx,
                cmd,
                self.radiance.image,
                color_range(),
                (S::COMPUTE_SHADER, A::SHADER_WRITE, L::GENERAL),
                (
                    S::FRAGMENT_SHADER,
                    A::SHADER_READ,
                    L::SHADER_READ_ONLY_OPTIMAL,
                ),
            );
            if !self.initialized {
                sync::transition(
                    &ctx,
                    cmd,
                    self.lut.image,
                    color_range(),
                    (S::COMPUTE_SHADER, A::SHADER_WRITE, L::GENERAL),
                    (
                        S::FRAGMENT_SHADER,
                        A::SHADER_READ,
                        L::SHADER_READ_ONLY_OPTIMAL,
                    ),
                );
            }
            sync::memory(
                &ctx,
                cmd,
                (S::COMPUTE_SHADER, A::SHADER_WRITE),
                (S::FRAGMENT_SHADER, A::SHADER_READ),
            );
        })?;
        self.initialized = true;
        tracing::info!(probe, width, height, "GPU IBL prefilter and SH complete");
        Ok(())
    }
    pub fn verify(&mut self, source: &EnvironmentRadiance) -> Result<[f32; 3], BoxError> {
        let (w, h) = source.dimensions();
        let reference = SpecularEnvironment::from_equirectangular(w, h, source.pixels())?;
        let atlas = read_image(
            self.memory.clone(),
            &self.radiance,
            SPECULAR_WIDTH,
            SPECULAR_HEIGHT * SPECULAR_LEVELS * PROBE_SLOTS,
        )?;
        let lut = read_image(self.memory.clone(), &self.lut, BRDF_LUT_SIZE, BRDF_LUT_SIZE)?;
        let error = |gpu: &[f32], cpu: &[[f32; 4]]| {
            gpu.iter()
                .zip(cpu.iter().flatten())
                .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                .fold(0.0_f32, f32::max)
        };
        let atlas_error = error(&atlas, reference.atlas());
        let lut_error = error(&lut, &govno_render::specular_brdf_lut());
        let ctx = self.memory.ctx.clone();
        Upload::run(ctx.clone(), |cmd| {
            sync::memory(
                &ctx,
                cmd,
                (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_WRITE,
                ),
                (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
            )
        })?;
        let sh = self.sh.read()?;
        // Match the GPU SH integration grid; the CPU full-resolution projection
        // intentionally uses different quadrature for small input panoramas.
        let samples: Vec<_> = (0..32)
            .flat_map(|y| {
                (0..64).map(move |x| {
                    let theta = std::f32::consts::PI * (y as f32 + 0.5) / 32.0;
                    let phi = std::f32::consts::TAU * (x as f32 + 0.5) / 64.0;
                    source.sample(glam::Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.cos(),
                        theta.sin() * phi.sin(),
                    ))
                })
            })
            .collect();
        let diffuse = govno_render::DiffuseEnvironment::from_equirectangular(64, 32, &samples)?;
        let cpu = diffuse.coefficients();
        let mut sh_error = 0.0_f32;
        for (i, c) in cpu.iter().enumerate() {
            for (channel, value) in c.to_array().into_iter().enumerate() {
                let offset = i * 16 + channel * 4;
                let gpu = f32::from_ne_bytes(sh[offset..offset + 4].try_into()?);
                sh_error = sh_error.max((gpu - value).abs() / value.abs().max(1.0));
            }
        }
        tracing::info!(
            atlas_error,
            lut_error,
            sh_error,
            "GPU IBL versus CPU reference"
        );
        if ![atlas_error, lut_error, sh_error]
            .iter()
            .all(|v| v.is_finite())
            || atlas_error > 0.04
            || lut_error > 0.01
            || sh_error > 0.01
        {
            return Err("GPU IBL reference tolerance exceeded".into());
        }
        Ok([atlas_error, lut_error, sh_error])
    }
    pub fn replace(&mut self, data: &SpecularEnvironment) -> Result<(), BoxError> {
        self.process(data.source(), 0)
    }
    pub fn bind(&self, sets: &[vk::DescriptorSet]) {
        for set in sets {
            let radiance = [vk::DescriptorImageInfo::default()
                .image_view(self.radiance.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let lut = [vk::DescriptorImageInfo::default()
                .image_view(self.lut.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let sampler = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
            let sh = [vk::DescriptorBufferInfo::default()
                .buffer(self.sh.handle)
                .range(PROBE_SLOTS as u64 * 9 * 16)];
            // SAFETY: caller supplies new/quiesced sets. All resources outlive submissions.
            unsafe {
                self.memory.ctx.device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(6)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&radiance),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(7)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .image_info(&sampler),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(8)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .image_info(&lut),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(9)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&sh),
                    ],
                    &[],
                );
            }
        }
    }
}
impl Drop for SpecularGpu {
    fn drop(&mut self) {
        // SAFETY: owner quiesces referring work before destruction.
        unsafe {
            self.memory.ctx.device.destroy_sampler(self.sampler, None);
        }
    }
}
fn upload(
    memory: Rc<Memory>,
    width: u32,
    height: u32,
    pixels: &[[f32; 4]],
) -> Result<ImageResource, BoxError> {
    let format = vk::Format::R32G32B32A32_SFLOAT;
    // SAFETY: physical device belongs to the retained instance.
    let features = unsafe {
        memory
            .ctx
            .instance()
            .get_physical_device_format_properties(memory.ctx.physical, format)
    }
    .optimal_tiling_features;
    if !features.contains(
        vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR
            | vk::FormatFeatureFlags::TRANSFER_DST
            | vk::FormatFeatureFlags::TRANSFER_SRC,
    ) {
        return Err(
            "RGBA32F sampled/linear-filter/transfer support required for specular IBL".into(),
        );
    }
    // SAFETY: physical device is retained by memory.ctx.
    let limits = unsafe {
        memory
            .ctx
            .instance()
            .get_physical_device_properties(memory.ctx.physical)
    }
    .limits;
    if width > limits.max_image_dimension2_d || height > limits.max_image_dimension2_d {
        return Err("environment dimensions exceed GPU image limits".into());
    }
    let image = ImageResource::sampled(memory.clone(), vk::Extent2D { width, height }, 1, format)?;
    let bytes: Vec<u8> = pixels
        .iter()
        .flatten()
        .flat_map(|v| v.to_ne_bytes())
        .collect();
    let mut staging = Buffer::new(
        memory.clone(),
        bytes.len() as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    )?;
    staging.write(&bytes)?;
    let ctx = memory.ctx.clone();
    Upload::run(ctx.clone(), |cmd| {
        // SAFETY: upload helper retains command/fence; image/staging are live until
        // completion. CPU cooker supplies exact dimensions and finite RGBA pixels.
        unsafe {
            let before = [vk::ImageMemoryBarrier2::default()
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image.image)
                .subresource_range(color_range())
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
            ctx.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&before),
            );
            let region = [vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })];
            ctx.device.cmd_copy_buffer_to_image(
                cmd,
                staging.handle,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );
            let after = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image.image)
                .subresource_range(color_range())
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
            ctx.device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&after),
            );
        }
    })?;
    Ok(image)
}

fn read_image(
    memory: Rc<Memory>,
    image: &ImageResource,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, BoxError> {
    let mut buffer = Buffer::new(
        memory.clone(),
        u64::from(width) * u64::from(height) * 16,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
    )?;
    let ctx = memory.ctx.clone();
    Upload::run(ctx.clone(), |cmd| {
        use vk::{AccessFlags2 as A, ImageLayout as L, PipelineStageFlags2 as S};
        sync::transition(
            &ctx,
            cmd,
            image.image,
            color_range(),
            (
                S::COMPUTE_SHADER | S::FRAGMENT_SHADER,
                A::SHADER_READ | A::SHADER_WRITE,
                L::SHADER_READ_ONLY_OPTIMAL,
            ),
            (S::COPY, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL),
        );
        let region = [vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })];
        // SAFETY: live image/buffer have matching extents, transfer usage and sufficient allocation.
        unsafe {
            ctx.device.cmd_copy_image_to_buffer(
                cmd,
                image.image,
                L::TRANSFER_SRC_OPTIMAL,
                buffer.handle,
                &region,
            );
        }
        sync::transition(
            &ctx,
            cmd,
            image.image,
            color_range(),
            (S::COPY, A::TRANSFER_READ, L::TRANSFER_SRC_OPTIMAL),
            (
                S::FRAGMENT_SHADER,
                A::SHADER_READ,
                L::SHADER_READ_ONLY_OPTIMAL,
            ),
        );
        sync::memory(
            &ctx,
            cmd,
            (S::COPY, A::TRANSFER_WRITE),
            (S::HOST, A::HOST_READ),
        );
    })?;
    Ok(buffer
        .read()?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect())
}

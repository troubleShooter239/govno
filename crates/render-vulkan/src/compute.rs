//! Small owned compute pipeline; no scene/render-graph abstraction.
use crate::context::DeviceContext;
use ash::vk;
use govno_core::BoxError;
use std::{ffi::CStr, io::Cursor, sync::Arc};
pub(crate) struct Compute {
    ctx: Arc<DeviceContext>,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
}
impl Compute {
    pub fn new(
        ctx: Arc<DeviceContext>,
        code: &[u8],
        entry: &CStr,
        bindings: &[(u32, vk::DescriptorType)],
        push: u32,
        count: u32,
    ) -> Result<Self, BoxError> {
        let mut owner = Self {
            ctx,
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            sets: vec![],
        };
        let bindings: Vec<_> = bindings
            .iter()
            .map(|&(binding, ty)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(ty)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let words = ash::util::read_spv(&mut Cursor::new(code))?;
        // SAFETY: create-info arrays remain live; every successful handle has an owner.
        unsafe {
            owner.descriptor_layout = owner.ctx.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sizes: Vec<_> = bindings
                .iter()
                .map(|b| {
                    vk::DescriptorPoolSize::default()
                        .ty(b.descriptor_type)
                        .descriptor_count(count)
                })
                .collect();
            owner.pool = owner.ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(count)
                    .pool_sizes(&sizes),
                None,
            )?;
            owner.sets = owner.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(owner.pool)
                    .set_layouts(&vec![owner.descriptor_layout; count as usize]),
            )?;
            owner.layout = owner.ctx.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[owner.descriptor_layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .size(push)]),
                None,
            )?;
            let module = owner
                .ctx
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
            let result = owner.ctx.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .layout(owner.layout)
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(entry),
                    )],
                None,
            );
            owner.ctx.device.destroy_shader_module(module, None);
            match result {
                Ok(p) => owner.pipeline = p[0],
                Err((partial, e)) => {
                    for p in partial {
                        owner.ctx.device.destroy_pipeline(p, None);
                    }
                    return Err(e.into());
                }
            }
        }
        Ok(owner)
    }
    pub fn dispatch(&self, cmd: vk::CommandBuffer, set: usize, push: &[u8], groups: [u32; 3]) {
        // SAFETY: caller records on a compute-capable queue and retains descriptor resources.
        unsafe {
            self.ctx
                .device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.ctx.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.sets[set]],
                &[],
            );
            self.ctx.device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
            self.ctx
                .device
                .cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
        }
    }
}
impl Drop for Compute {
    fn drop(&mut self) {
        // SAFETY: users quiesce work before destruction; partial construction has no users.
        unsafe {
            self.ctx.device.destroy_pipeline(self.pipeline, None);
            self.ctx.device.destroy_pipeline_layout(self.layout, None);
            self.ctx.device.destroy_descriptor_pool(self.pool, None);
            self.ctx
                .device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

use govno_core::{BoxError, ModuleId};
use govno_runtime::{Engine, EngineBuilder, EngineModule, ModuleRegistration, RunningModule};

struct DemoModule {
    id: ModuleId,
    dependencies: &'static [ModuleId],
}
struct DemoState(ModuleId);
impl RunningModule for DemoState {
    fn shutdown(&mut self) -> Result<(), BoxError> {
        tracing::info!(module = %self.0, "demo resources released");
        Ok(())
    }
}
impl EngineModule for DemoModule {
    fn register(&self, builder: &mut EngineBuilder) {
        let id = self.id;
        builder.register(ModuleRegistration::new(
            id,
            self.dependencies.iter().copied(),
            move || {
                tracing::info!(module = %id, "starting demo module");
                Ok(Box::new(DemoState(id)))
            },
        ));
    }
}
fn main() -> Result<(), BoxError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .try_init()?;
    // Deliberately register the dependent first to exercise dependency ordering.
    let engine = Engine::builder()
        .with_module(DemoModule {
            id: ModuleId("demo-scene"),
            dependencies: &[ModuleId("demo-assets")],
        })
        .with_module(DemoModule {
            id: ModuleId("demo-assets"),
            dependencies: &[],
        })
        .build()?;
    engine.run(|| {
        tracing::info!("Foundation sandbox ready; graphics arrive in Vulkan Bootstrap");
        Ok(())
    })?;
    Ok(())
}

//! Explicit, single-threaded lifecycle orchestration.
//! Registration may be Send + Sync; running resources deliberately need not be.
use govno_core::{BoxError, ModuleId};
use std::{collections::HashSet, fmt};

pub trait EngineModule: Send + Sync {
    fn register(&self, builder: &mut EngineBuilder);
}

/// Owns resources of one successfully initialized subsystem.
/// Implementations must release partial initialization through RAII if their
/// factory returns Err. shutdown is called once; Drop remains a final backstop.
pub trait RunningModule {
    fn shutdown(&mut self) -> Result<(), BoxError>;
}

type Factory = Box<dyn FnOnce() -> Result<Box<dyn RunningModule>, BoxError>>;

pub struct ModuleRegistration {
    id: ModuleId,
    dependencies: Vec<ModuleId>,
    factory: Factory,
}

impl ModuleRegistration {
    pub fn new(
        id: ModuleId,
        dependencies: impl IntoIterator<Item = ModuleId>,
        factory: impl FnOnce() -> Result<Box<dyn RunningModule>, BoxError> + 'static,
    ) -> Self {
        Self {
            id,
            dependencies: dependencies.into_iter().collect(),
            factory: Box::new(factory),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    Duplicate(ModuleId),
    MissingDependency {
        module: ModuleId,
        dependency: ModuleId,
    },
    /// Contains all unresolved modules, including any dependents of a cycle.
    DependencyCycle {
        blocked: Vec<ModuleId>,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid module graph: {self:?}")
    }
}
impl std::error::Error for BuildError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Startup,
    Application,
    Shutdown,
}

#[derive(Debug)]
pub struct Failure {
    pub module: Option<ModuleId>,
    pub phase: Phase,
    pub source: BoxError,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} ({:?}): {}", self.phase, self.module, self.source)
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// First failure is the primary cause; subsequent failures are cleanup errors.
#[derive(Debug)]
pub struct RunError {
    pub failures: Vec<Failure>,
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, failure) in self.failures.iter().enumerate() {
            if index > 0 {
                write!(f, "; ")?;
            }
            write!(f, "{failure}")?;
        }
        Ok(())
    }
}
impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.failures.first().map(|f| f as &dyn std::error::Error)
    }
}

#[derive(Default)]
pub struct EngineBuilder {
    registrations: Vec<ModuleRegistration>,
}

impl EngineBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_module(mut self, module: impl EngineModule) -> Self {
        module.register(&mut self);
        self
    }

    pub fn register(&mut self, registration: ModuleRegistration) {
        self.registrations.push(registration);
    }

    /// Validates the entire graph before invoking any resource factory.
    /// Independent modules preserve registration order.
    pub fn build(self) -> Result<Engine, BuildError> {
        let mut ids = HashSet::new();
        for registration in &self.registrations {
            if !ids.insert(registration.id) {
                return Err(BuildError::Duplicate(registration.id));
            }
        }
        for registration in &self.registrations {
            for dependency in &registration.dependencies {
                if !ids.contains(dependency) {
                    return Err(BuildError::MissingDependency {
                        module: registration.id,
                        dependency: *dependency,
                    });
                }
            }
        }
        let mut pending = self.registrations;
        let mut ready = HashSet::new();
        let mut ordered = Vec::new();
        while !pending.is_empty() {
            let Some(index) = pending
                .iter()
                .position(|r| r.dependencies.iter().all(|d| ready.contains(d)))
            else {
                return Err(BuildError::DependencyCycle {
                    blocked: pending.iter().map(|r| r.id).collect(),
                });
            };
            let registration = pending.remove(index);
            ready.insert(registration.id);
            ordered.push(registration);
        }
        Ok(Engine { ordered })
    }
}

pub struct Engine {
    ordered: Vec<ModuleRegistration>,
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    pub fn module_order(&self) -> impl Iterator<Item = ModuleId> + '_ {
        self.ordered.iter().map(|r| r.id)
    }

    /// Runs one application lifetime, then tears down all initialized modules.
    /// The caller owns the application/event loop. A real window loop is stage 1.
    pub fn run(self, application: impl FnOnce() -> Result<(), BoxError>) -> Result<(), RunError> {
        let mut active = ActiveModules::default();
        let mut failures = Vec::new();
        for registration in self.ordered {
            let id = registration.id;

            match (registration.factory)() {
                Ok(module) => active.0.push((id, module)),
                Err(source) => {
                    failures.push(Failure {
                        module: Some(id),
                        phase: Phase::Startup,
                        source,
                    });
                    failures.extend(active.shutdown());
                    return Err(RunError { failures });
                }
            }
        }
        if let Err(source) = application() {
            failures.push(Failure {
                module: None,
                phase: Phase::Application,
                source,
            });
        }
        failures.extend(active.shutdown());
        if failures.is_empty() {
            Ok(())
        } else {
            Err(RunError { failures })
        }
    }
}

#[derive(Default)]
struct ActiveModules(Vec<(ModuleId, Box<dyn RunningModule>)>);

impl ActiveModules {
    fn shutdown(&mut self) -> Vec<Failure> {
        let mut failures = Vec::new();
        while let Some((id, mut module)) = self.0.pop() {
            if let Err(source) = module.shutdown() {
                failures.push(Failure {
                    module: Some(id),
                    phase: Phase::Shutdown,
                    source,
                });
            }
        }
        failures
    }
}

impl Drop for ActiveModules {
    fn drop(&mut self) {
        // Best-effort cleanup during application/factory unwinding.
        // shutdown implementations must not panic.
        for failure in self.shutdown() {
            eprintln!("module cleanup failed during unwind: {failure}");
        }
    }
}

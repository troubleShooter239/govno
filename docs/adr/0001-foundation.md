# ADR 0001: Foundation and module boundaries

Date: 2026-10-26. Status: implemented initial foundation.
Source: decisions recovered from the engine-plan chat and the user's answers.
This English revision repairs unreadable text in the initial documentation.

## Constraints

Rust Stable, Edition 2024; Cargo Workspace; statically linked modules.
Vulkan 1.3 minimum with 1.4 preferred; Windows 11/Linux x86-64.
Primary hardware validation is on RTX 4080 Super under Windows.
Linux validation is deferred by the user.
The playable MVP is a small first-person scene with movement, camera,
lighting and physical objects. No editor, runtime DLL/SO plugins, mobile
targets or universal graphics RHI in the MVP.

## Foundation

core contains minimal shared contracts; runtime depends on core.
EngineModule: Send + Sync registers module factories in EngineBuilder.
Factories and RunningModule do not require Send/Sync; main-thread-only
resources are supported without unsafe thread assertions.

build() validates duplicate IDs, missing dependencies and cycles before
creating resources. A deterministic topological order chooses the earliest
registered ready module. A cycle diagnostic lists blocked modules, which
can also include downstream dependents.

run() initializes modules, executes the application closure and shuts down
initialized modules in reverse order. The closure owns the application loop.
This is a lifetime boundary, not a scheduler or service locator.

Each successful factory transfers resource ownership to RunningModule.
A failing factory must clean its own partially initialized resources via RAII.
Startup failure prevents later modules from starting and rolls back earlier ones.
Shutdown errors do not stop the cleanup of remaining modules.

BuildError reports graph problems. RunError retains the primary error and all
cleanup errors, including underlying error sources. Panics are not converted
to normal errors; unwind triggers best-effort cleanup. shutdown/Drop must not
panic. panic=abort cannot guarantee cleanup.

## Architecture rules

Core does not depend on Vulkan, ECS or physics. Renderer does not inspect game
ECS components. Scene extraction will produce RenderWorld in a later phase.
Assets will load CPU data without depending on a window or Vulkan device.
Game code must not manipulate VkImage/VkBuffer directly.

unsafe is forbidden in ordinary crates. render-vulkan owns audited FFI blocks.
Rust Drop alone does not prove GPU completion. Backend destruction requires
explicit completion tracking; later dynamic resources need deferred deletion.

## Validation

11 lifecycle tests cover ordering, graph errors, partial startup, shutdown
errors, application errors, empty engines, unstarted engines and unwind.
The tests use Rc-backed resources to verify main-thread state compatibility.

Windows CI is configured; remote CI has not been run from this session.
Service access and scheduling remain future work, not an implicit DI framework.
The license is still undecided.

## References

- https://doc.rust-lang.org/cargo/reference/workspaces.html
- https://doc.rust-lang.org/stable/edition-guide/rust-2024/cargo-resolver.html
- https://docs.rs/tracing-subscriber/latest/tracing_subscriber/fmt/

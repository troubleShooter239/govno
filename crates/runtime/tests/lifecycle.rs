use govno_core::{BoxError, ModuleId};
use govno_runtime::{BuildError, EngineBuilder, ModuleRegistration, Phase, RunningModule};
use std::{cell::RefCell, rc::Rc};

type Log = Rc<RefCell<Vec<String>>>;
struct State {
    id: &'static str,
    log: Log,
    fail_stop: bool,
}
impl RunningModule for State {
    fn shutdown(&mut self) -> Result<(), BoxError> {
        self.log.borrow_mut().push(format!("stop:{}", self.id));
        if self.fail_stop {
            Err(std::io::Error::other(self.id).into())
        } else {
            Ok(())
        }
    }
}
fn add(
    builder: &mut EngineBuilder,
    log: &Log,
    id: &'static str,
    deps: &[&'static str],
    fail_start: bool,
    fail_stop: bool,
) {
    let log = log.clone();
    builder.register(ModuleRegistration::new(
        ModuleId(id),
        deps.iter().map(|d| ModuleId(d)),
        move || {
            log.borrow_mut().push(format!("start:{id}"));
            if fail_start {
                return Err(std::io::Error::other(id).into());
            }
            Ok(Box::new(State { id, log, fail_stop }))
        },
    ));
}
#[test]
fn dependencies_start_first_and_stop_last() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "scene", &["assets"], false, false);
    add(&mut b, &log, "assets", &[], false, false);
    add(&mut b, &log, "audio", &[], false, false);
    let engine = b.build().unwrap();
    assert_eq!(
        engine.module_order().collect::<Vec<_>>(),
        vec![ModuleId("assets"), ModuleId("scene"), ModuleId("audio")]
    );
    engine
        .run(|| {
            log.borrow_mut().push("app".into());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        *log.borrow(),
        [
            "start:assets",
            "start:scene",
            "start:audio",
            "app",
            "stop:audio",
            "stop:scene",
            "stop:assets"
        ]
    );
}
#[test]
fn duplicate_rejected_before_initialization() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &[], false, false);
    add(&mut b, &log, "a", &[], false, false);
    assert!(matches!(
        b.build(),
        Err(BuildError::Duplicate(ModuleId("a")))
    ));
    assert!(log.borrow().is_empty());
}
#[test]
fn missing_dependency_rejected_before_initialization() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &["missing"], false, false);
    assert!(matches!(
        b.build(),
        Err(BuildError::MissingDependency {
            module: ModuleId("a"),
            dependency: ModuleId("missing")
        })
    ));
    assert!(log.borrow().is_empty());
}
#[test]
fn cycles_and_self_dependencies_rejected() {
    for self_cycle in [false, true] {
        let log = Log::default();
        let mut b = EngineBuilder::new();
        add(
            &mut b,
            &log,
            "a",
            if self_cycle { &["a"] } else { &["b"] },
            false,
            false,
        );
        if !self_cycle {
            add(&mut b, &log, "b", &["a"], false, false);
        }
        assert!(matches!(b.build(), Err(BuildError::DependencyCycle { .. })));
        assert!(log.borrow().is_empty());
    }
}
#[test]
fn failed_start_rolls_back_and_preserves_cleanup_errors() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &[], false, true);
    add(&mut b, &log, "b", &["a"], true, false);
    add(&mut b, &log, "c", &["b"], false, false);
    let error = b
        .build()
        .unwrap()
        .run(|| panic!("must not run"))
        .unwrap_err();
    assert_eq!(error.failures.len(), 2);
    assert_eq!(error.failures[0].phase, Phase::Startup);
    assert_eq!(error.failures[0].module, Some(ModuleId("b")));
    assert_eq!(error.failures[1].phase, Phase::Shutdown);
    assert_eq!(*log.borrow(), ["start:a", "start:b", "stop:a"]);
}
#[test]
fn application_failure_still_stops_every_module() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &[], false, true);
    add(&mut b, &log, "b", &["a"], false, true);
    let error = b
        .build()
        .unwrap()
        .run(|| Err(std::io::Error::other("app").into()))
        .unwrap_err();
    assert_eq!(error.failures.len(), 3);
    assert_eq!(error.failures[0].phase, Phase::Application);
    assert_eq!(error.failures[1].module, Some(ModuleId("b")));
    assert_eq!(error.failures[2].module, Some(ModuleId("a")));
    assert_eq!(*log.borrow(), ["start:a", "start:b", "stop:b", "stop:a"]);
}
#[test]
fn shutdown_failure_is_returned_after_successful_application() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &[], false, true);
    let error = b.build().unwrap().run(|| Ok(())).unwrap_err();
    assert_eq!(error.failures.len(), 1);
    assert_eq!(error.failures[0].phase, Phase::Shutdown);
}
#[test]
fn empty_engine_runs_application() {
    let mut ran = false;
    EngineBuilder::new()
        .build()
        .unwrap()
        .run(|| {
            ran = true;
            Ok(())
        })
        .unwrap();
    assert!(ran);
}
#[test]
fn dropping_unstarted_engine_does_not_initialize_resources() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &[], false, false);
    drop(b.build().unwrap());
    assert!(log.borrow().is_empty());
}
#[test]
fn application_unwind_cleans_up_started_modules() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &[], false, false);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = b.build().unwrap().run(|| panic!("application panic"));
    }));
    assert!(result.is_err());
    assert_eq!(*log.borrow(), ["start:a", "stop:a"]);
}
#[test]
fn factory_unwind_cleans_up_previously_started_modules() {
    let log = Log::default();
    let mut b = EngineBuilder::new();
    add(&mut b, &log, "a", &[], false, false);
    b.register(ModuleRegistration::new(
        ModuleId("b"),
        [ModuleId("a")],
        || panic!("factory panic"),
    ));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = b.build().unwrap().run(|| Ok(()));
    }));
    assert!(result.is_err());
    assert_eq!(*log.borrow(), ["start:a", "stop:a"]);
}

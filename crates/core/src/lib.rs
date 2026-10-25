//! Small shared contracts. No dependency on ECS, graphics, or platform APIs.
use std::fmt;

/// Stable module identity. Names are unique within an engine instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub &'static str);

impl fmt::Display for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Errors crossing subsystem boundaries retain their original source.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

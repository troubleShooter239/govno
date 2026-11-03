//! Single-queue pass scheduling, resource hazards and lifetime planning.
//! State is supplied by the backend; the graph owns no GPU objects.
use std::{
    collections::BTreeSet,
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceId {
    owner: u64,
    index: usize,
}
impl ResourceId {
    pub fn index(self) -> usize {
        self.index
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassId {
    owner: u64,
    index: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}
impl Access {
    fn reads(self) -> bool {
        self != Self::Write
    }
    fn writes(self) -> bool {
        self != Self::Read
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Use<S> {
    pub resource: ResourceId,
    pub state: S,
    pub access: Access,
}
struct Resource<S> {
    name: String,
    initial: S,
    initialized: bool,
}
struct Pass<S, T> {
    name: String,
    tag: T,
    uses: Vec<Use<S>>,
    after: Vec<PassId>,
}
pub struct Graph<S, T> {
    owner: u64,
    resources: Vec<Resource<S>>,
    passes: Vec<Pass<S, T>>,
}
impl<S, T> Default for Graph<S, T> {
    fn default() -> Self {
        Self {
            owner: NEXT_GRAPH
                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .expect("graph ID exhausted"),
            resources: Vec::new(),
            passes: Vec::new(),
        }
    }
}
#[derive(Debug, PartialEq, Eq)]
pub enum GraphError {
    InvalidResource(String),
    InvalidDependency(String),
    DuplicateUse(String),
    UninitializedRead { pass: String, resource: String },
    Cycle,
}
impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid render graph: {self:?}")
    }
}
impl std::error::Error for GraphError {}

#[derive(Debug, Clone)]
pub struct Transition<S> {
    pub resource: ResourceId,
    pub before: S,
    pub after: S,
    /// Last writer and subsequent readers. Retain writer visibility when a later
    /// read uses a different stage, and all reader scopes for write-after-read.
    pub sources: Vec<S>,
}
pub struct CompiledPass<S, T> {
    pub name: String,
    pub tag: T,
    pub transitions: Vec<Transition<S>>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub first: usize,
    pub last: usize,
}
pub struct CompiledGraph<S, T> {
    pub passes: Vec<CompiledPass<S, T>>,
    lifetimes: Vec<Option<Lifetime>>,
    owner: u64,
}
impl<S, T> CompiledGraph<S, T> {
    pub fn lifetime(&self, id: ResourceId) -> Option<Lifetime> {
        (id.owner == self.owner)
            .then(|| self.lifetimes.get(id.index).copied().flatten())
            .flatten()
    }
}
impl<S: Copy + Eq, T> Graph<S, T> {
    /// Uninitialized resources must be written before any read, including ReadWrite.
    pub fn resource(
        &mut self,
        name: impl Into<String>,
        initial: S,
        initialized: bool,
    ) -> ResourceId {
        let id = ResourceId {
            owner: self.owner,
            index: self.resources.len(),
        };
        self.resources.push(Resource {
            name: name.into(),
            initial,
            initialized,
        });
        id
    }
    pub fn pass(&mut self, name: impl Into<String>, tag: T, uses: Vec<Use<S>>) -> PassId {
        let id = PassId {
            owner: self.owner,
            index: self.passes.len(),
        };
        self.passes.push(Pass {
            name: name.into(),
            tag,
            uses,
            after: Vec::new(),
        });
        id
    }
    pub fn after(&mut self, pass: PassId, dependency: PassId) -> Result<(), GraphError> {
        if pass.owner != self.owner
            || dependency.owner != self.owner
            || pass.index >= self.passes.len()
            || dependency.index >= self.passes.len()
        {
            return Err(GraphError::InvalidDependency(
                "foreign or missing pass".into(),
            ));
        }
        self.passes[pass.index].after.push(dependency);
        Ok(())
    }
    /// Uses are declared in resource-version order; explicit edges order side effects.
    /// RAW, WAR and WAW hazards all become edges. Compilation is deterministic.
    pub fn compile(self) -> Result<CompiledGraph<S, T>, GraphError> {
        let n = self.passes.len();
        let r = self.resources.len();
        let mut edges = vec![BTreeSet::new(); n];
        let mut writers: Vec<Option<usize>> = vec![None; r];
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); r];
        for (i, pass) in self.passes.iter().enumerate() {
            for dep in &pass.after {
                edges[dep.index].insert(i);
            }
            let mut seen = BTreeSet::new();
            for usage in &pass.uses {
                let id = usage.resource;
                if id.owner != self.owner || id.index >= r {
                    return Err(GraphError::InvalidResource(pass.name.clone()));
                }
                if !seen.insert(id.index) {
                    return Err(GraphError::DuplicateUse(pass.name.clone()));
                }
                if let Some(writer) = writers[id.index] {
                    edges[writer].insert(i);
                }
                if usage.access.writes() {
                    for reader in readers[id.index].drain(..) {
                        edges[reader].insert(i);
                    }
                    writers[id.index] = Some(i);
                } else {
                    readers[id.index].push(i);
                }
            }
        }
        let mut indegree = vec![0; n];
        for successors in &edges {
            for &next in successors {
                indegree[next] += 1;
            }
        }
        let mut ready: BTreeSet<_> = (0..n).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(i) = ready.pop_first() {
            order.push(i);
            for &next in &edges[i] {
                indegree[next] -= 1;
                if indegree[next] == 0 {
                    ready.insert(next);
                }
            }
        }
        if order.len() != n {
            return Err(GraphError::Cycle);
        }
        let mut states: Vec<_> = self.resources.iter().map(|r| r.initial).collect();
        let mut initialized: Vec<_> = self.resources.iter().map(|r| r.initialized).collect();
        let mut previous_write = vec![false; r];
        let mut sources: Vec<Vec<S>> = states.iter().map(|&s| vec![s]).collect();
        let mut lifetimes: Vec<Option<Lifetime>> = vec![None; r];
        let mut pending: Vec<_> = self.passes.into_iter().map(Some).collect();
        let mut passes = Vec::with_capacity(n);
        for (position, i) in order.into_iter().enumerate() {
            let pass = pending[i].take().expect("unique topological index");
            let mut transitions = Vec::new();
            for usage in pass.uses {
                let k = usage.resource.index;
                if usage.access.reads() && !initialized[k] {
                    return Err(GraphError::UninitializedRead {
                        pass: pass.name,
                        resource: self.resources[k].name.clone(),
                    });
                }
                if lifetimes[k].is_none()
                    || states[k] != usage.state
                    || previous_write[k]
                    || usage.access.writes()
                {
                    transitions.push(Transition {
                        resource: usage.resource,
                        before: states[k],
                        after: usage.state,
                        sources: sources[k].clone(),
                    });
                }
                states[k] = usage.state;
                if usage.access.writes() {
                    sources[k].clear();
                }
                if !sources[k].contains(&usage.state) {
                    sources[k].push(usage.state);
                }
                previous_write[k] = usage.access.writes();
                initialized[k] |= usage.access.writes();
                match &mut lifetimes[k] {
                    Some(l) => l.last = position,
                    None => {
                        lifetimes[k] = Some(Lifetime {
                            first: position,
                            last: position,
                        })
                    }
                }
            }
            passes.push(CompiledPass {
                name: pass.name,
                tag: pass.tag,
                transitions,
            });
        }
        Ok(CompiledGraph {
            passes,
            lifetimes,
            owner: self.owner,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hazards_retain_writer_and_all_readers_before_overwrite() {
        let mut g = Graph::default();
        let image = g.resource("image", 0, false);
        g.pass(
            "write",
            0,
            vec![Use {
                resource: image,
                state: 1,
                access: Access::Write,
            }],
        );
        g.pass(
            "compute read",
            1,
            vec![Use {
                resource: image,
                state: 2,
                access: Access::Read,
            }],
        );
        g.pass(
            "fragment read",
            2,
            vec![Use {
                resource: image,
                state: 3,
                access: Access::Read,
            }],
        );
        g.pass(
            "overwrite",
            3,
            vec![Use {
                resource: image,
                state: 1,
                access: Access::Write,
            }],
        );
        let plan = g.compile().unwrap();
        assert_eq!(
            plan.passes.iter().map(|p| p.tag).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        let fragment = &plan.passes[2].transitions[0];
        assert_eq!(fragment.before, 2);
        assert_eq!(fragment.sources, [1, 2]);
        let overwrite = &plan.passes[3].transitions[0];
        assert_eq!(overwrite.sources, [1, 2, 3]);
        assert_eq!(plan.lifetime(image), Some(Lifetime { first: 0, last: 3 }));
    }
    #[test]
    fn detects_cycles_and_uninitialized_reads() {
        let mut g = Graph::default();
        let r = g.resource("uninitialized", 0, false);
        g.pass(
            "read",
            (),
            vec![Use {
                resource: r,
                state: 1,
                access: Access::Read,
            }],
        );
        assert!(matches!(
            g.compile(),
            Err(GraphError::UninitializedRead { .. })
        ));
        let mut g = Graph::<u8, _>::default();
        let a = g.pass("a", (), vec![]);
        let b = g.pass("b", (), vec![]);
        g.after(a, b).unwrap();
        g.after(b, a).unwrap();
        assert!(matches!(g.compile(), Err(GraphError::Cycle)));
    }
    #[test]
    fn independent_passes_reorder_but_foreign_resources_are_rejected() {
        let mut g = Graph::<u8, _>::default();
        let a = g.pass("a", 0, vec![]);
        let b = g.pass("b", 1, vec![]);
        g.after(a, b).unwrap();
        assert_eq!(
            g.compile()
                .unwrap()
                .passes
                .iter()
                .map(|p| p.tag)
                .collect::<Vec<_>>(),
            [1, 0]
        );
        let mut other = Graph::<u8, ()>::default();
        let foreign = other.resource("foreign", 0, true);
        let mut g = Graph::default();
        g.resource("local", 0, true);
        g.pass(
            "bad",
            (),
            vec![Use {
                resource: foreign,
                state: 1,
                access: Access::Read,
            }],
        );
        assert!(matches!(g.compile(), Err(GraphError::InvalidResource(_))));
        let mut g = Graph::default();
        let imported = g.resource("external contents", 7, true);
        g.pass(
            "first read",
            (),
            vec![Use {
                resource: imported,
                state: 7,
                access: Access::Read,
            }],
        );
        let plan = g.compile().unwrap();
        assert_eq!(plan.passes[0].transitions[0].sources, [7]);
    }
}

//! Physical input and configurable gameplay actions. No window, ECS or GPU types.
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputKey {
    W,
    A,
    S,
    D,
    Q,
    E,
    Shift,
    Space,
    F,
    R,
    F1,
    F5,
}

#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    Key { key: InputKey, pressed: bool },
    Look { x: f32, y: f32 },
    FocusLost,
}

/// Application-defined identifier; games can declare additional action constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Action(pub &'static str);

pub mod actions {
    use super::Action;
    pub const FORWARD: Action = Action("forward");
    pub const BACKWARD: Action = Action("backward");
    pub const LEFT: Action = Action("left");
    pub const RIGHT: Action = Action("right");
    pub const RISE: Action = Action("rise");
    pub const DESCEND: Action = Action("descend");
    pub const SPRINT: Action = Action("sprint");
    pub const JUMP: Action = Action("jump");
    pub const INTERACT: Action = Action("interact");
    pub const RESET: Action = Action("reset");
    pub const DEBUG: Action = Action("debug");
    pub const RELOAD_SCENE: Action = Action("reload_scene");
}

#[derive(Debug, Clone, Default)]
pub struct ActionMap {
    bindings: HashMap<Action, HashSet<InputKey>>,
}

impl ActionMap {
    pub fn bind(mut self, action: Action, key: InputKey) -> Self {
        self.bindings.entry(action).or_default().insert(key);
        self
    }

    pub fn desktop() -> Self {
        use actions::*;
        Self::default()
            .bind(FORWARD, InputKey::W)
            .bind(BACKWARD, InputKey::S)
            .bind(LEFT, InputKey::A)
            .bind(RIGHT, InputKey::D)
            .bind(RISE, InputKey::E)
            .bind(DESCEND, InputKey::Q)
            .bind(SPRINT, InputKey::Shift)
            .bind(JUMP, InputKey::Space)
            .bind(INTERACT, InputKey::F)
            .bind(RESET, InputKey::R)
            .bind(DEBUG, InputKey::F1)
            .bind(RELOAD_SCENE, InputKey::F5)
    }
}

pub struct InputState {
    map: ActionMap,
    held: HashSet<InputKey>,
    pressed: HashSet<Action>,
    released: HashSet<Action>,
    look: [f32; 2],
}

impl InputState {
    pub fn new(map: ActionMap) -> Self {
        Self {
            map,
            held: HashSet::new(),
            pressed: HashSet::new(),
            released: HashSet::new(),
            look: [0.0; 2],
        }
    }

    /// Rebinding resets pending physical state, including pointer deltas.
    pub fn set_map(&mut self, map: ActionMap) {
        *self = Self::new(map);
    }

    pub fn process(&mut self, event: InputEvent) {
        match event {
            InputEvent::Key { key, pressed } => {
                // Releasing one binding must not release an action still held by another.
                for (&action, keys) in &self.map.bindings {
                    if !keys.contains(&key) {
                        continue;
                    }
                    let before = keys.iter().any(|k| self.held.contains(k));
                    let after = keys.iter().any(|k| {
                        if *k == key {
                            pressed
                        } else {
                            self.held.contains(k)
                        }
                    });
                    if !before && after {
                        self.pressed.insert(action);
                    }
                    if before && !after {
                        self.released.insert(action);
                    }
                }
                if pressed {
                    self.held.insert(key);
                } else {
                    self.held.remove(&key);
                }
            }
            InputEvent::Look { x, y } => {
                let next = [self.look[0] + x, self.look[1] + y];
                if next.iter().all(|v| v.is_finite()) {
                    self.look = next;
                }
            }
            InputEvent::FocusLost => {
                self.pressed.clear();
                for (&action, keys) in &self.map.bindings {
                    if keys.iter().any(|k| self.held.contains(k)) {
                        self.released.insert(action);
                    }
                }
                self.held.clear();
                self.look = [0.0; 2];
            }
        }
    }

    pub fn held(&self, action: Action) -> bool {
        self.map
            .bindings
            .get(&action)
            .is_some_and(|keys| keys.iter().any(|key| self.held.contains(key)))
    }

    pub fn just_pressed(&self, action: Action) -> bool {
        self.pressed.contains(&action)
    }
    pub fn just_released(&self, action: Action) -> bool {
        self.released.contains(&action)
    }

    pub fn axis(&self, negative: Action, positive: Action) -> f32 {
        u8::from(self.held(positive)) as f32 - u8::from(self.held(negative)) as f32
    }

    /// Mouse distances are consumed once per render update, never scaled by dt.
    pub fn take_look(&mut self) -> [f32; 2] {
        std::mem::take(&mut self.look)
    }

    /// Call after each fixed tick, not each frame. Edges survive frames without
    /// a tick and cannot repeat across catch-up ticks.
    pub fn finish_tick(&mut self) {
        self.pressed.clear();
        self.released.clear();
    }
}

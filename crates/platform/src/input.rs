#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputKey {
    W,
    A,
    S,
    D,
    Q,
    E,
    Shift,
}
#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    Key { key: InputKey, pressed: bool },
    Look { x: f32, y: f32 },
    FocusLost,
}

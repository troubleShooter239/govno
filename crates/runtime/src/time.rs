//! Bounded fixed-step simulation clock, independent of presentation cadence.
use govno_core::BoxError;
use std::time::Duration;

const MAX_FRAME: Duration = Duration::from_millis(250);

pub struct FixedClock {
    step: Duration,
    max_steps: u32,
    accumulator: Duration,
}

#[derive(Debug, Clone, Copy)]
pub struct FixedUpdate {
    pub steps: u32,
    /// Interpolate previous -> current state for rendering; always in [0, 1).
    pub alpha: f32,
    /// Wall time intentionally discarded to bound work after a stall/suspension.
    pub discarded: Duration,
}

impl FixedClock {
    pub fn new(step: Duration, max_steps: u32) -> Result<Self, BoxError> {
        if step.is_zero() || step > MAX_FRAME || !(1..=1024).contains(&max_steps) {
            return Err("fixed step must be in (0,250ms], with 1..=1024 catch-up steps".into());
        }
        Ok(Self {
            step,
            max_steps,
            accumulator: Duration::ZERO,
        })
    }

    pub fn step(&self) -> Duration {
        self.step
    }

    /// Discards accumulated wall time when pausing/unloading a scene.
    pub fn reset(&mut self) {
        self.accumulator = Duration::ZERO;
    }

    pub fn advance(&mut self, elapsed: Duration) -> FixedUpdate {
        let accepted = elapsed.min(MAX_FRAME);
        self.accumulator += accepted;
        // At most 500ms of nanoseconds, so the conversion is bounded.
        let available = (self.accumulator.as_nanos() / self.step.as_nanos()) as u64;
        let steps = available.min(u64::from(self.max_steps)) as u32;
        let remainder =
            Duration::from_nanos((self.accumulator.as_nanos() % self.step.as_nanos()) as u64);
        let discarded = elapsed - accepted + (self.accumulator - remainder - self.step * steps);
        self.accumulator = remainder;
        FixedUpdate {
            steps,
            // f64 -> f32 rounding must not turn a near-full remainder into 1.
            alpha: ((remainder.as_secs_f64() / self.step.as_secs_f64()) as f32)
                .min(f32::from_bits(1.0_f32.to_bits() - 1)),
            discarded,
        }
    }
}

impl Default for FixedClock {
    fn default() -> Self {
        Self {
            step: Duration::from_nanos(16_666_667),
            max_steps: 8,
            accumulator: Duration::ZERO,
        }
    }
}

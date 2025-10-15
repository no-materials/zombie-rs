//! Walk lifecycle instrumentation.
//!
//! This module defines the lightweight event types that higher-level code can
//! subscribe to in order to monitor Monte Carlo walks. Observers receive
//! notifications for walk start, intermediate steps, and termination, enabling
//! features such as live visualisation, debugging, or statistical summaries.

use alloc::vec::Vec;

use crate::math::Vec3;

/// Reason a walk terminated.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TerminationReason {
    /// The walk reached a boundary point within the target tolerance.
    HitBoundary,
    /// The walk exceeded the configured step budget.
    MaxSteps,
}

/// Snapshot emitted when a walk begins.
#[derive(Copy, Clone, Debug)]
pub struct WalkStart {
    /// Initial position inside the domain.
    pub position: Vec3,
}

/// Snapshot emitted for every accepted step inside a walk.
#[derive(Copy, Clone, Debug)]
pub struct WalkStep {
    /// Position before the jump.
    pub position: Vec3,
    /// Radius of the largest empty ball centred at `position`.
    pub radius: f32,
    /// Zero-based index of the step.
    pub depth: u32,
}

/// Snapshot emitted when a walk finishes.
#[derive(Copy, Clone, Debug)]
pub struct WalkTerminate {
    /// The sampled boundary point (for `HitBoundary`) or the last position (for `MaxSteps`).
    pub position: Vec3,
    /// Why the walk stopped.
    pub reason: TerminationReason,
    /// Number of steps performed (zero-based index of last step, or 0 if none).
    pub depth: u32,
}

/// Final state returned by walk evaluators.
#[derive(Copy, Clone, Debug)]
pub struct WalkOutcome {
    /// Estimated solution value accumulated along the walk.
    pub value: f32,
    /// Termination reason observed.
    pub reason: TerminationReason,
    /// Number of steps performed during the final attempt.
    pub steps: u32,
}

impl WalkOutcome {
    /// Convenience constructor.
    pub const fn new(value: f32, reason: TerminationReason, steps: u32) -> Self {
        Self { value, reason, steps }
    }
}

/// Observer interface for receiving walk events.
pub trait WalkObserver: Send + Sync {
    /// Called before the first step of a walk.
    fn on_start(&self, _event: WalkStart) {}
    /// Called for every accepted step.
    fn on_step(&self, _event: WalkStep) {}
    /// Called once when the walk terminates.
    fn on_terminate(&self, _event: WalkTerminate) {}
}

/// Observer implementation that does nothing; used when no instrumentation is requested.
pub(crate) struct NoopObserver;

impl WalkObserver for NoopObserver {}

/// Helper that fans out notifications to many observers.
pub(crate) struct ObserverList<'a> {
    observers: Vec<&'a dyn WalkObserver>,
}

impl<'a> ObserverList<'a> {
    /// Build a fan-out observer from an iterator of borrowed observers.
    pub fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a dyn WalkObserver>,
    {
        Self {
            observers: iter.into_iter().collect(),
        }
    }
}

impl<'a> WalkObserver for ObserverList<'a> {
    fn on_start(&self, event: WalkStart) {
        for obs in &self.observers {
            obs.on_start(event);
        }
    }

    fn on_step(&self, event: WalkStep) {
        for obs in &self.observers {
            obs.on_step(event);
        }
    }

    fn on_terminate(&self, event: WalkTerminate) {
        for obs in &self.observers {
            obs.on_terminate(event);
        }
    }
}

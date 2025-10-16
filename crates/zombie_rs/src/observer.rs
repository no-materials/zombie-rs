//! Walk lifecycle instrumentation.
//!
//! This module defines the lightweight event types that higher-level code can
//! subscribe to in order to monitor Monte Carlo walks. Observers receive
//! notifications for walk start, intermediate steps, and termination, enabling
//! features such as live visualisation, debugging, or statistical summaries.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::Write;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use spin::Mutex;

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
        Self {
            value,
            reason,
            steps,
        }
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

/// Observer implementation that does nothing; useful when no instrumentation is requested.
pub struct NoopObserver;

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

/// Snapshot of aggregate walk statistics.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WalkStatsSnapshot {
    /// Total number of walks observed.
    pub walks: u32,
    /// Number of walks that terminated by hitting the boundary tolerance.
    pub boundary_hits: u32,
    /// Number of walks that exhausted the step budget.
    pub max_steps_hits: u32,
    /// Total number of steps across all walks.
    pub total_steps: u64,
}

struct StatsInner {
    walks: AtomicU32,
    boundary_hits: AtomicU32,
    max_steps_hits: AtomicU32,
    total_steps: AtomicU64,
}

impl StatsInner {
    const fn new() -> Self {
        Self {
            walks: AtomicU32::new(0),
            boundary_hits: AtomicU32::new(0),
            max_steps_hits: AtomicU32::new(0),
            total_steps: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> WalkStatsSnapshot {
        WalkStatsSnapshot {
            walks: self.walks.load(Ordering::Relaxed),
            boundary_hits: self.boundary_hits.load(Ordering::Relaxed),
            max_steps_hits: self.max_steps_hits.load(Ordering::Relaxed),
            total_steps: self.total_steps.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.walks.store(0, Ordering::Relaxed);
        self.boundary_hits.store(0, Ordering::Relaxed);
        self.max_steps_hits.store(0, Ordering::Relaxed);
        self.total_steps.store(0, Ordering::Relaxed);
    }
}

unsafe impl Send for StatsInner {}
unsafe impl Sync for StatsInner {}

/// Thread-friendly accumulator that tracks aggregate walk statistics using atomics.
#[derive(Clone)]
pub struct StatsObserver {
    inner: Arc<StatsInner>,
}

impl StatsObserver {
    /// Create a fresh statistics accumulator.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(StatsInner::new()),
        }
    }

    /// Snapshot the current statistics.
    pub fn snapshot(&self) -> WalkStatsSnapshot {
        self.inner.snapshot()
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.inner.reset();
    }
}

impl Default for StatsObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl WalkObserver for StatsObserver {
    fn on_start(&self, _event: WalkStart) {
        self.inner.walks.fetch_add(1, Ordering::Relaxed);
    }

    fn on_step(&self, _event: WalkStep) {
        self.inner.total_steps.fetch_add(1, Ordering::Relaxed);
    }

    fn on_terminate(&self, event: WalkTerminate) {
        match event.reason {
            TerminationReason::HitBoundary => {
                self.inner.boundary_hits.fetch_add(1, Ordering::Relaxed);
            }
            TerminationReason::MaxSteps => {
                self.inner.max_steps_hits.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum PlyPointKind {
    Start,
    Step,
    Hit,
    MaxSteps,
}

impl PlyPointKind {
    fn color(self) -> (u8, u8, u8) {
        match self {
            PlyPointKind::Start => (0, 200, 255),  // cyan
            PlyPointKind::Step => (255, 255, 255), // white
            PlyPointKind::Hit => (0, 255, 0),      // green
            PlyPointKind::MaxSteps => (255, 0, 0), // red
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct PlyPoint {
    position: Vec3,
    color: (u8, u8, u8),
}

struct PlyInner {
    // Simple spin mutex keeps the implementation correct; revisit with lock-free structure if contention becomes an issue.
    points: Mutex<Vec<PlyPoint>>,
}

impl PlyInner {
    fn new() -> Self {
        Self {
            points: Mutex::new(Vec::new()),
        }
    }

    fn clear(&self) {
        self.points.lock().clear();
    }

    fn push(&self, point: PlyPoint) {
        self.points.lock().push(point);
    }

    fn snapshot(&self) -> Vec<PlyPoint> {
        self.points.lock().clone()
    }
}

/// Observer that records walk events into a coloured PLY point cloud.
/// Use [`PlyRecorder::to_ascii`] to obtain a textual `.ply` file.                                                                                                                                                                                                         
#[derive(Clone)]
pub struct PlyRecorder {
    inner: Arc<PlyInner>,
}
impl PlyRecorder {
    /// Create an empty recorder.                                                                                                                                                                                                                                          
    pub fn new() -> Self {
        Self {
            inner: Arc::new(PlyInner::new()),
        }
    }

    /// Remove all stored points.                                                                                                                                                                                                                                          
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Export the recorded walks as an ASCII PLY file with per-point RGB colours.
    pub fn to_ascii(&self) -> String {
        let points = self.inner.snapshot();
        let mut out = String::new();
        let _ = writeln!(out, "ply");
        let _ = writeln!(out, "format ascii 1.0");
        let _ = writeln!(out, "element vertex {}", points.len());
        out.push_str("property float x\nproperty float y\nproperty float z\n");
        out.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
        out.push_str("end_header\n");

        for p in points {
            let (r, g, b) = p.color;
            let _ = writeln!(
                out,
                "{:.6} {:.6} {:.6} {} {} {}",
                p.position.x, p.position.y, p.position.z, r, g, b,
            );
        }
        out
    }
}

impl Default for PlyRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl WalkObserver for PlyRecorder {
    fn on_start(&self, event: WalkStart) {
        self.inner.push(PlyPoint {
            position: event.position,
            color: PlyPointKind::Start.color(),
        });
    }

    fn on_step(&self, event: WalkStep) {
        self.inner.push(PlyPoint {
            position: event.position,
            color: PlyPointKind::Step.color(),
        });
    }

    fn on_terminate(&self, event: WalkTerminate) {
        let kind = match event.reason {
            TerminationReason::HitBoundary => PlyPointKind::Hit,
            TerminationReason::MaxSteps => PlyPointKind::MaxSteps,
        };
        self.inner.push(PlyPoint {
            position: event.position,
            color: kind.color(),
        });
    }
}

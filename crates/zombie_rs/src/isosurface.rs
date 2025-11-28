//! Progressive isosurface scaffolding.
//!
//! This module defines the data structures and scheduling hooks needed to
//! incrementally extract an isosurface using Monte Carlo samples, without
//! performing meshing yet. The intent is to progressively refine an octree of
//! cells ordered by variance or view importance and let later stages plug in
//! Walk-on-Spheres evaluations and meshing.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::marker::PhantomData;

use crate::math::{Aabb, Vec3};
use crate::params::WalkBudget;
use crate::solver::Solver;
use crate::stats::Stats;
use crate::{BoundaryDirichlet, ClosestAccel, Domain, SourceTerm};

/// Per-cell statistics and bounds for progressive sampling.
#[derive(Clone, Debug)]
pub struct Cell {
    /// Minimum corner of the axis-aligned bounds.
    bbox_min: Vec3,
    /// Maximum corner of the axis-aligned bounds.
    bbox_max: Vec3,
    /// Per-corner streaming statistics of the sampled field.
    samples: [Stats; 8],
    /// Cached variance proxy for queue priority.
    variance: f32,
    /// Octree depth (root = 0).
    depth: u8,
    /// Child cells in Morton order; `None` when not yet subdivided.
    #[allow(dead_code)]
    children: [Option<Box<Cell>>; 8],
    /// Logical timestamp used by schedulers to track recency.
    last_touched: u64,
}

impl Cell {
    /// Create a new leaf cell covering `bbox_min..bbox_max` at `depth`.
    pub fn new(bbox_min: Vec3, bbox_max: Vec3, depth: u8) -> Self {
        Self {
            bbox_min,
            bbox_max,
            samples: core::array::from_fn(|_| Stats::default()),
            variance: 0.0,
            depth,
            children: core::array::from_fn(|_| None),
            last_touched: 0,
        }
    }

    /// Axis-aligned bounding box of the cell.
    pub fn bbox(&self) -> Aabb {
        Aabb {
            min: self.bbox_min,
            max: self.bbox_max,
        }
    }

    /// Center of the cell.
    pub fn center(&self) -> Vec3 {
        (self.bbox_min + self.bbox_max) * 0.5
    }

    /// Minimum corner of the bounds.
    pub fn bbox_min(&self) -> Vec3 {
        self.bbox_min
    }

    /// Maximum corner of the bounds.
    pub fn bbox_max(&self) -> Vec3 {
        self.bbox_max
    }

    /// Update the cached variance using the maximum corner variance.
    pub fn refresh_variance(&mut self) {
        self.variance = self.samples.iter().fold(0.0_f32, |acc, s| acc.max(s.var()));
    }

    /// Return `true` when the cell can be split.
    pub fn can_subdivide(&self, max_depth: u8) -> bool {
        self.depth < max_depth
    }

    /// Split the cell into eight children; each child starts with empty stats.
    pub fn subdivide(&self) -> [Cell; 8] {
        let mid = self.center();
        let min = self.bbox_min;
        let max = self.bbox_max;
        core::array::from_fn(|i| {
            let child_min = Vec3::new(
                if i & 1 == 0 { min.x } else { mid.x },
                if i & 2 == 0 { min.y } else { mid.y },
                if i & 4 == 0 { min.z } else { mid.z },
            );
            let child_max = Vec3::new(
                if i & 1 == 0 { mid.x } else { max.x },
                if i & 2 == 0 { mid.y } else { max.y },
                if i & 4 == 0 { mid.z } else { max.z },
            );
            Cell::new(child_min, child_max, self.depth.saturating_add(1))
        })
    }

    /// Read the cached variance used for prioritisation.
    pub fn variance(&self) -> f32 {
        self.variance
    }

    /// Override the cached variance (used internally by schedulers).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_variance(&mut self, variance: f32) {
        self.variance = variance;
    }

    /// Depth of this cell in the octree (root = 0).
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// Update the depth (used internally; tests craft tie-break scenarios).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_depth(&mut self, depth: u8) {
        self.depth = depth;
    }

    /// Last-touched logical timestamp.
    pub fn last_touched(&self) -> u64 {
        self.last_touched
    }

    /// Update the last-touched timestamp.
    #[allow(dead_code)]
    pub(crate) fn set_last_touched(&mut self, stamp: u64) {
        self.last_touched = stamp;
    }

    /// Immutable view of the corner statistics.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn samples(&self) -> &[Stats; 8] {
        &self.samples
    }

    /// Mutable view of the corner statistics.
    #[allow(dead_code)]
    pub(crate) fn samples_mut(&mut self) -> &mut [Stats; 8] {
        &mut self.samples
    }

    /// Immutable view of child handles.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn children(&self) -> &[Option<Box<Cell>>; 8] {
        &self.children
    }

    /// Mutable view of child handles.
    #[allow(dead_code)]
    pub(crate) fn children_mut(&mut self) -> &mut [Option<Box<Cell>>; 8] {
        &mut self.children
    }
}

/// User-facing knobs for progressive sampling.
#[derive(Copy, Clone, Debug)]
pub struct IsoParams {
    /// Target iso-value to track.
    pub iso_value: f32,
    /// Variance threshold used to decide refinement.
    pub variance_tol: f32,
    /// Maximum octree depth.
    pub max_depth: u8,
    /// Number of samples to take when a cell is processed.
    pub batch_samples: u32,
    /// Walk configuration forwarded to WoS estimators.
    pub walk: WalkBudget,
}

impl IsoParams {
    /// Construct a parameter set with explicit variance tolerance and depth.
    pub fn new(iso_value: f32, variance_tol: f32, max_depth: u8, walk: WalkBudget) -> Self {
        Self {
            iso_value,
            variance_tol,
            max_depth,
            batch_samples: 1,
            walk,
        }
    }

    /// Override the per-cell batch size.
    pub fn with_batch_samples(self, batch_samples: u32) -> Self {
        Self {
            batch_samples: batch_samples.max(1),
            ..self
        }
    }
}

/// Incremental mesh delta placeholder (meshing arrives in later PRs).
#[derive(Clone, Debug, Default)]
pub struct MeshDelta {
    /// Vertex positions emitted by a scheduler iteration.
    pub vertices: Vec<Vec3>,
    /// Triangle indices emitted by a scheduler iteration.
    pub indices: Vec<[u32; 3]>,
}

/// Scheduler responsible for ordering cells and dispatching sampling batches.
///
/// This stub wires the domain and PDE types so later PRs can plug in WoS calls.
pub struct IsoScheduler<'a, D, A, G, F>
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    F: SourceTerm,
{
    /// Global sampling parameters.
    params: IsoParams,
    /// Storage for all known cells (indexed by queue entries).
    cells: Vec<Cell>,
    /// Priority queue of cell indices.
    queue: BinaryHeap<QueuedCell>,
    /// Monotonic counter to break priority ties.
    seq: u64,
    /// Marker tying the scheduler to the PDE traits without yet invoking them.
    _marker: PhantomData<(&'a Solver<'a, D, A>, &'a G, &'a F)>,
}

impl<'a, D, A, G, F> IsoScheduler<'a, D, A, G, F>
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    F: SourceTerm,
{
    /// Create a scheduler seeded with a root cell.
    pub fn new(params: IsoParams, root: Cell) -> Self {
        let mut cells = Vec::new();
        let queue = BinaryHeap::new();
        cells.push(root);
        let mut sched = Self {
            params,
            cells,
            queue,
            seq: 0,
            _marker: PhantomData,
        };
        sched.push_index(0);
        sched
    }

    /// Pop the highest-priority cell index.
    pub fn next_cell(&mut self) -> Option<usize> {
        self.queue.pop().map(|entry| entry.cell_index)
    }

    /// Access an immutable view of a cell by index.
    pub fn cell(&self, index: usize) -> Option<&Cell> {
        self.cells.get(index)
    }

    /// Access a mutable view of a cell by index.
    pub fn cell_mut(&mut self, index: usize) -> Option<&mut Cell> {
        self.cells.get_mut(index)
    }

    /// Placeholder for future WoS sampling; currently just drains the queue.
    ///
    /// The intention is that later patches will perform sampling, update stats,
    /// and requeue the cell or its children. Returning `None` indicates no mesh
    /// output is produced yet.
    pub fn step(&mut self) -> Option<MeshDelta> {
        let _ = self.params;
        self.next_cell().map(|_| MeshDelta::default())
    }

    /// Push children derived from a parent cell into storage and queue.
    pub fn enqueue_children(&mut self, children: &[Cell; 8]) -> [usize; 8] {
        core::array::from_fn(|i| {
            let idx = self.cells.len();
            self.cells.push(children[i].clone());
            self.push_index(idx);
            idx
        })
    }

    fn push_index(&mut self, index: usize) {
        let variance = self.cells.get(index).map(|c| c.variance).unwrap_or(0.0_f32);
        let key = PriorityKey::new(variance, self.cells[index].depth, self.seq);
        self.seq = self.seq.saturating_add(1);
        self.queue.push(QueuedCell {
            key,
            cell_index: index,
        });
    }
}

/// Ordering key for the priority queue (max-heap on variance, then depth, then FIFO).
#[derive(Copy, Clone, Debug, PartialEq)]
struct PriorityKey {
    /// Cached variance used as the primary priority.
    variance: f32,
    /// Depth used to break ties (shallower first).
    depth: u8,
    /// Sequence number used to enforce FIFO within identical priority.
    seq: u64,
}

impl PriorityKey {
    /// Create a key with explicit fields; variance must be finite.
    fn new(variance: f32, depth: u8, seq: u64) -> Self {
        Self {
            variance,
            depth,
            seq,
        }
    }
}

impl Eq for PriorityKey {}

impl Ord for PriorityKey {
    fn cmp(&self, other: &Self) -> Ordering {
        match self
            .variance
            .partial_cmp(&other.variance)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => match self.depth.cmp(&other.depth) {
                Ordering::Equal => self.seq.cmp(&other.seq),
                // Prefer shallower nodes first.
                ord => ord.reverse(),
            },
            // BinaryHeap is a max-heap; keep higher variance first.
            ord => ord,
        }
    }
}

impl PartialOrd for PriorityKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Heap payload holding a cell index and its priority.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct QueuedCell {
    /// Priority key.
    key: PriorityKey,
    /// Index into `IsoScheduler::cells`.
    cell_index: usize,
}

impl Ord for QueuedCell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl PartialOrd for QueuedCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a unit cube root cell.
    fn root_cell() -> Cell {
        Cell::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 0)
    }

    #[test]
    fn priority_queue_orders_by_variance_then_depth() {
        let mut a = root_cell();
        a.set_variance(0.1);
        let mut b = root_cell();
        b.set_variance(0.5);
        let mut c = root_cell();
        c.set_variance(0.5);
        c.set_depth(2);
        a.set_last_touched(42);
        assert_eq!(a.last_touched(), 42);
        assert_eq!(a.samples().len(), 8);
        assert_eq!(a.children().len(), 8);

        let params = IsoParams::new(0.0, 0.01, 4, WalkBudget::new(1e-3, 16));
        type TD = crate::SdfDomain<fn(Vec3) -> f32>;
        type TA = crate::ClosestNaive;
        type TB = crate::BoundaryDirichletFn<fn(Vec3) -> f32>;
        type TS = crate::PointSource;
        let mut sched: IsoScheduler<'static, TD, TA, TB, TS> = IsoScheduler::new(params, a);

        // Push extra cells manually to test ordering.
        sched.cells.push(b);
        sched.push_index(1);
        sched.cells.push(c);
        sched.push_index(2);

        let first = sched.next_cell().unwrap();
        let second = sched.next_cell().unwrap();
        let third = sched.next_cell().unwrap();

        assert_eq!(first, 1, "highest variance should pop first");
        assert_eq!(
            second, 2,
            "equal variance prefers shallower depth; shallower (b) was popped first, so deeper pops next"
        );
        assert_eq!(third, 0, "lowest variance pops last");
    }

    #[test]
    fn cell_subdivide_splits_bounds() {
        let cell = root_cell();
        let children = cell.subdivide();
        for (i, child) in children.iter().enumerate() {
            assert_eq!(child.depth(), 1);
            let mid = cell.center();
            let min = child.bbox_min();
            let max = child.bbox_max();
            // Each child spans half the parent in every dimension.
            assert!(
                (max.x - min.x - 0.5).abs() < 1e-6,
                "child {i} x extent wrong"
            );
            assert!(
                (max.y - min.y - 0.5).abs() < 1e-6,
                "child {i} y extent wrong"
            );
            assert!(
                (max.z - min.z - 0.5).abs() < 1e-6,
                "child {i} z extent wrong"
            );
            // Midpoint placement sanity: child min/max must align to parent min/mid/max.
            assert!(
                (min.x - cell.bbox_min().x).abs() < 1e-6
                    || (min.x - mid.x).abs() < 1e-6
                    || (min.x - cell.bbox_max().x).abs() < 1e-6
            );
            assert!(
                (max.x - cell.bbox_min().x).abs() < 1e-6
                    || (max.x - mid.x).abs() < 1e-6
                    || (max.x - cell.bbox_max().x).abs() < 1e-6
            );
        }
    }
}

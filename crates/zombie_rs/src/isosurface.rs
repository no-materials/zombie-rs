//! Progressive isosurface scaffolding.
//!
//! This module implements the sampling core for progressive isosurface extraction.
//! Cells are refined based on variance or detected sign-changes of the sampled
//! field, using Monte Carlo estimators from the existing `Solver` API. A small
//! mesher is provided that consumes converged cells, but meshing is kept
//! decoupled from scheduling so sampling can be driven independently.

extern crate alloc;

use alloc::boxed::Box;
use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::marker::PhantomData;

use crate::math::{Aabb, Vec3};
use crate::params::{GradParams, PoissonParams, WalkBudget};
use crate::rng::Rng;
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
    /// Streaming statistics for the cell center (used when enabled).
    center: Stats,
    /// Optional running mean gradients per corner.
    corner_grad: [Option<Vec3>; 8],
    /// Optional running mean gradient at the cell center.
    center_grad: Option<Vec3>,
    /// Deterministic RNG bound to this cell for reproducible sampling.
    rng: Rng,
    /// Cached variance proxy for queue priority.
    variance: f32,
    /// Octree depth (root = 0).
    depth: u8,
    /// Child cells in Morton order; `None` when not yet subdivided.
    children: [Option<Box<Cell>>; 8],
    /// Logical timestamp used by schedulers to track recency.
    last_touched: u64,
}

impl Cell {
    /// Create a new leaf cell covering `bbox_min..bbox_max` at `depth` with a deterministic seed.
    pub fn new(bbox_min: Vec3, bbox_max: Vec3, depth: u8, seed: u64) -> Self {
        Self {
            bbox_min,
            bbox_max,
            samples: core::array::from_fn(|_| Stats::default()),
            center: Stats::default(),
            corner_grad: core::array::from_fn(|_| None),
            center_grad: None,
            rng: Rng::seed_from(seed),
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

    /// Corner positions in Morton order (000..111).
    pub fn corner_positions(&self) -> [Vec3; 8] {
        let min = self.bbox_min;
        let max = self.bbox_max;
        [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(min.x, max.y, max.z),
            Vec3::new(max.x, max.y, max.z),
        ]
    }

    /// Update the cached variance using both corner and center statistics.
    pub fn refresh_variance(&mut self) {
        self.variance = self
            .samples
            .iter()
            .fold(self.center.var(), |acc, s| acc.max(s.var()));
    }

    /// Return `true` when the cell has been sampled at least once.
    fn has_samples(&self, include_center: bool) -> bool {
        let mut seen = self.samples.iter().any(|s| s.count() > 0);
        if include_center {
            seen |= self.center.count() > 0;
        }
        seen
    }

    /// Return `true` when sampling has converged or the cell cannot be subdivided.
    ///
    /// Convergence requires at least one sample; a cell at `max_depth` is also
    /// treated as converged even if its variance is still above tolerance because
    /// no further refinement is possible.
    pub fn is_converged(&self, params: &IsoParams) -> bool {
        if !self.has_samples(params.sample_center) {
            return false;
        }
        self.variance() <= params.variance_tol || !self.can_subdivide(params.max_depth)
    }

    /// Return `true` when the cell can be split.
    pub fn can_subdivide(&self, max_depth: u8) -> bool {
        self.depth < max_depth
    }

    /// Split the cell into eight children seeded with `seeds`; each child starts empty.
    pub fn subdivide_with_seeds(&self, seeds: [u64; 8]) -> [Cell; 8] {
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
            Cell::new(child_min, child_max, self.depth.saturating_add(1), seeds[i])
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
    pub(crate) fn samples(&self) -> &[Stats; 8] {
        &self.samples
    }

    /// Copy out corner means for meshing or diagnostics.
    pub(crate) fn corner_means(&self) -> [f32; 8] {
        core::array::from_fn(|i| self.samples[i].mean())
    }

    /// Mutable view of the corner statistics.
    pub(crate) fn samples_mut(&mut self) -> &mut [Stats; 8] {
        &mut self.samples
    }

    /// Immutable view of child handles.
    #[allow(dead_code)]
    pub(crate) fn children(&self) -> &[Option<Box<Cell>>; 8] {
        &self.children
    }

    /// Mutable view of child handles.
    pub(crate) fn children_mut(&mut self) -> &mut [Option<Box<Cell>>; 8] {
        &mut self.children
    }

    /// Immutable view of the center statistics.
    pub fn center_stats(&self) -> &Stats {
        &self.center
    }

    /// Mutable view of the center statistics.
    pub(crate) fn center_stats_mut(&mut self) -> &mut Stats {
        &mut self.center
    }

    /// Mutable access to the cell RNG for sampling.
    fn rng_mut(&mut self) -> &mut Rng {
        &mut self.rng
    }

    /// Re-seed the cell RNG deterministically.
    pub(crate) fn reset_rng(&mut self, seed: u64) {
        self.rng = Rng::seed_from(seed);
    }

    /// Mutable view of corner gradients.
    pub(crate) fn corner_grads_mut(&mut self) -> &mut [Option<Vec3>; 8] {
        &mut self.corner_grad
    }

    /// Copy out corner gradients (if any have been accumulated).
    pub(crate) fn corner_grads(&self) -> [Option<Vec3>; 8] {
        self.corner_grad
    }

    /// Mutable view of the center gradient.
    pub(crate) fn center_grad_mut(&mut self) -> &mut Option<Vec3> {
        &mut self.center_grad
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
    /// Poisson parameters forwarded to estimators.
    pub poisson: PoissonParams,
    /// Optional gradient sampling configuration.
    pub grad: Option<GradParams>,
    /// Whether to sample the cell center in addition to corners.
    pub sample_center: bool,
    /// Global base seed used to derive per-cell RNG seeds.
    pub base_seed: u64,
}

impl IsoParams {
    /// Construct a parameter set with explicit variance tolerance and depth.
    pub fn new(
        iso_value: f32,
        variance_tol: f32,
        max_depth: u8,
        walk: WalkBudget,
        poisson: PoissonParams,
    ) -> Self {
        Self {
            iso_value,
            variance_tol,
            max_depth,
            batch_samples: 1,
            walk,
            poisson,
            grad: None,
            sample_center: true,
            base_seed: 0xA5A5_A5A5_1234_5678,
        }
    }

    /// Override the per-cell batch size.
    pub fn with_batch_samples(self, batch_samples: u32) -> Self {
        Self {
            batch_samples: batch_samples.max(1),
            ..self
        }
    }

    /// Enable gradient sampling.
    pub fn with_grad(self, grad: GradParams) -> Self {
        Self {
            grad: Some(grad),
            ..self
        }
    }

    /// Disable center sampling (corners only).
    pub fn without_center_sampling(self) -> Self {
        Self {
            sample_center: false,
            ..self
        }
    }

    /// Override the global base seed.
    pub fn with_base_seed(self, base_seed: u64) -> Self {
        Self { base_seed, ..self }
    }
}

/// Incremental mesh delta emitted per-cell by the mesher/scheduler.
#[derive(Clone, Debug, Default)]
pub struct MeshDelta {
    /// Vertex positions emitted by a scheduler iteration.
    pub vertices: Vec<Vec3>,
    /// Triangle indices emitted by a scheduler iteration.
    pub indices: Vec<[u32; 3]>,
}

/// Lightweight mesher that turns converged cells into triangle batches.
///
/// This struct is intentionally decoupled from the scheduler: callers pass in
/// precomputed corner samples (and optional gradients) and receive a `MeshDelta`
/// scoped to a single cell. Vertex de-duplication across cells is left to the
/// caller to keep the mesher stateless and easily testable. Internally this uses
/// marching tetrahedra (6 tets per cube) to avoid the large Marching Cubes table
/// while still producing watertight meshes for most configurations.
pub struct Mesher {
    /// Iso-value to contour.
    iso: f32,
    /// Toggle for gradient snapping along edges.
    use_gradient_snap: bool,
}

impl Mesher {
    /// Create a mesher that extracts the iso-surface `iso`. Gradient snap is enabled by default.
    pub fn new(iso: f32) -> Self {
        Self {
            iso,
            use_gradient_snap: true,
        }
    }

    /// Disable gradient-based snapping of edge intersections.
    pub fn without_gradient_snap(self) -> Self {
        Self {
            use_gradient_snap: false,
            ..self
        }
    }

    /// Generate a mesh for a single cell using marching tetrahedra.
    ///
    /// The `corners` array is expected in Morton order (000..111). When gradients
    /// are provided, a single Newton-style step is blended with linear interpolation
    /// to tighten edge intersections; otherwise pure linear interpolation is used.
    pub fn mesh_cell(
        &self,
        corners: [f32; 8],
        gradients: Option<[Option<Vec3>; 8]>,
        positions: [Vec3; 8],
    ) -> MeshDelta {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for tet in TETS {
            // Build local values/positions for this tet.
            let mut tv = [0f32; 4];
            let mut tp = [Vec3::new(0.0, 0.0, 0.0); 4];
            let mut tg: [Option<Vec3>; 4] = [None, None, None, None];
            for (i, &cidx) in tet.iter().enumerate() {
                tv[i] = corners[cidx];
                tp[i] = positions[cidx];
                if let Some(allg) = gradients.as_ref() {
                    tg[i] = allg[cidx];
                }
            }

            let mask = self.tet_mask(&tv);
            for tri in Self::tet_tris(mask) {
                let mut idx = [0u32; 3];
                for (k, &edge_id) in tri.iter().enumerate() {
                    let (c0, c1) = TET_EDGES[edge_id as usize];
                    let p = self.edge_point(tp[c0], tv[c0], tg[c0], tp[c1], tv[c1], tg[c1]);
                    idx[k] = vertices.len() as u32;
                    vertices.push(p);
                }
                indices.push(idx);
            }
        }

        MeshDelta { vertices, indices }
    }

    /// Compute the tetrahedron mask for marching tetrahedra.
    fn tet_mask(&self, values: &[f32; 4]) -> u8 {
        let mut mask = 0u8;
        for (i, &v) in values.iter().enumerate() {
            if v > self.iso {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Return triangle edge ids for a given tetrahedron mask (0..15), using symmetry to cover all cases.
    fn tet_tris(mask: u8) -> &'static [[i8; 3]] {
        let idx = (mask & 0x0F) as usize;
        // Cases above 7 mirror the inside/outside assignment; reuse complements.
        if idx <= 7 {
            &TET_TRI_TABLE_POS[idx]
        } else {
            &TET_TRI_TABLE_NEG[15 - idx]
        }
    }

    /// Interpolate an edge intersection, optionally blending a gradient snap.
    fn edge_point(
        &self,
        p0: Vec3,
        v0: f32,
        g0: Option<Vec3>,
        p1: Vec3,
        v1: f32,
        g1: Option<Vec3>,
    ) -> Vec3 {
        let iso = self.iso;
        let t_lin = ((iso - v0) / (v1 - v0 + 1e-8)).clamp(0.0, 1.0);
        let p_lin = p0 + (p1 - p0) * t_lin;

        if !self.use_gradient_snap {
            return p_lin;
        }

        // Use the gradient from the nearer corner if available.
        let (ref_point, ref_value, grad_opt) = if (iso - v0).abs() < (iso - v1).abs() {
            (p0, v0, g0)
        } else {
            (p1, v1, g1)
        };

        if let Some(g) = grad_opt {
            let denom = g.dot(g).max(1e-12);
            let step = (ref_value - iso) / denom;
            let p_newton = ref_point - g * step;
            // Blend to avoid overshoot; 0.5 is a pragmatic default.
            return p_lin * 0.5 + p_newton * 0.5;
        }

        p_lin
    }
}

/// Scheduler responsible for ordering cells and dispatching sampling batches.
pub struct IsoScheduler<'a, D, A, G, F>
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    F: SourceTerm,
{
    /// Global sampling parameters.
    params: IsoParams,
    /// Borrowed solver used for value/gradient estimates.
    solver: &'a Solver<'a, D, A>,
    /// Dirichlet boundary data.
    boundary: &'a G,
    /// Volume source term.
    source: &'a F,
    /// Storage for all known cells (indexed by queue entries).
    cells: Vec<Cell>,
    /// Priority queue of cell indices.
    queue: BinaryHeap<QueuedCell>,
    /// Monotonic counter to break priority ties.
    seq: u64,
    /// Marker tying the scheduler to the PDE traits.
    _marker: PhantomData<(&'a Solver<'a, D, A>, &'a G, &'a F)>,
}

/// Refinement and convergence flags derived after sampling a cell.
#[derive(Copy, Clone, Debug)]
struct StepFlags {
    /// Whether the current samples straddle the iso-value.
    sign_change: bool,
    /// Whether the cached variance exceeds tolerance.
    variance_high: bool,
    /// Whether the cell can still be subdivided.
    can_subdivide: bool,
    /// Whether the cell is converged (variance low or depth capped).
    converged: bool,
}

impl<'a, D, A, G, F> IsoScheduler<'a, D, A, G, F>
where
    D: Domain,
    A: ClosestAccel<D>,
    G: BoundaryDirichlet,
    F: SourceTerm,
{
    /// Create a scheduler seeded with a root cell.
    pub fn new(
        params: IsoParams,
        mut root: Cell,
        solver: &'a Solver<'a, D, A>,
        boundary: &'a G,
        source: &'a F,
    ) -> Self {
        root.reset_rng(params.base_seed);
        let mut cells = Vec::new();
        let queue = BinaryHeap::new();
        cells.push(root);
        let mut sched = Self {
            params,
            solver,
            boundary,
            source,
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

    /// Perform one scheduler iteration: sample a cell, update statistics, and refine or requeue.
    /// Also mesh converged cells using `mesher`.
    pub fn step(&mut self, mesher: &Mesher) -> Option<MeshDelta> {
        let idx = self.next_cell()?;
        let flags = self.sample_and_flags(idx);

        let subdivided = if (flags.sign_change || flags.variance_high) && flags.can_subdivide {
            self.spawn_children(idx);
            true
        } else {
            false
        };

        if !flags.converged {
            if flags.variance_high && !subdivided {
                self.push_index(idx);
            } else if flags.sign_change && !subdivided && !flags.variance_high {
                self.push_index(idx);
            }
        }

        let mesh = if flags.converged {
            self.mesh_ready_cell(idx, mesher)
        } else {
            MeshDelta::default()
        };

        Some(mesh)
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

    /// Push a cell index into the priority queue based on its cached variance.
    fn push_index(&mut self, index: usize) {
        let variance = self
            .cells
            .get(index)
            .map(|c| c.variance())
            .unwrap_or(0.0_f32);
        let key = PriorityKey::new(variance, self.cells[index].depth(), self.seq);
        self.seq = self.seq.saturating_add(1);
        self.queue.push(QueuedCell {
            key,
            cell_index: index,
        });
    }

    /// Sample a cell and compute refinement flags in one pass.
    fn sample_and_flags(&mut self, idx: usize) -> StepFlags {
        // Borrow target cell mutably without aliasing the vector.
        let (_, tail) = self.cells.split_at_mut(idx);
        let cell = tail.first_mut().expect("queued index must exist");
        let params = self.params;
        let solver = self.solver;
        let boundary = self.boundary;
        let source = self.source;

        Self::sample_cell(cell, params, solver, boundary, source);
        cell.refresh_variance();

        let sign = Self::has_sign_change(cell, params.iso_value, params.sample_center);
        let variance_high = cell.variance() > params.variance_tol;
        let can_subdivide = cell.can_subdivide(params.max_depth);
        let converged = cell.is_converged(&params);

        StepFlags {
            sign_change: sign,
            variance_high,
            can_subdivide,
            converged,
        }
    }

    /// Sample all corners (and optionally center) once per batch iteration.
    fn sample_cell(
        cell: &mut Cell,
        params: IsoParams,
        solver: &Solver<'a, D, A>,
        boundary: &G,
        source: &F,
    ) {
        let corners = cell.corner_positions();
        for _ in 0..params.batch_samples {
            for (i, p) in corners.iter().enumerate() {
                let val = Self::sample_value(cell, *p, params, solver, boundary, source);
                cell.samples_mut()[i].push(val);
                if let Some(gradp) = params.grad {
                    let g = Self::sample_grad(cell, *p, gradp, params, solver, boundary, source);
                    let count = cell.samples()[i].count().max(1);
                    Self::accumulate_vec3(&mut cell.corner_grads_mut()[i], count, g);
                }
            }

            if params.sample_center {
                let center_pos = cell.center();
                let val = Self::sample_value(cell, center_pos, params, solver, boundary, source);
                cell.center_stats_mut().push(val);
                if let Some(gradp) = params.grad {
                    let g = Self::sample_grad(
                        cell, center_pos, gradp, params, solver, boundary, source,
                    );
                    let count = cell.center_stats().count().max(1);
                    Self::accumulate_vec3(cell.center_grad_mut(), count, g);
                }
            }
        }
    }

    /// Single value sample via the Poisson Dirichlet estimator.
    fn sample_value(
        cell: &mut Cell,
        p: Vec3,
        params: IsoParams,
        solver: &Solver<'a, D, A>,
        boundary: &G,
        source: &F,
    ) -> f32 {
        solver.poisson_dirichlet(
            boundary,
            source,
            params.walk,
            params.poisson,
            cell.rng_mut(),
            p,
        )
    }

    /// Optional gradient sample via the Poisson gradient estimator.
    fn sample_grad(
        cell: &mut Cell,
        p: Vec3,
        grad: GradParams,
        params: IsoParams,
        solver: &Solver<'a, D, A>,
        boundary: &G,
        source: &F,
    ) -> Vec3 {
        solver.poisson_gradient(
            boundary,
            source,
            params.walk,
            params.poisson,
            grad,
            cell.rng_mut(),
            p,
        )
    }

    /// Detect whether the current statistics cross the iso-value.
    fn has_sign_change(cell: &Cell, iso: f32, include_center: bool) -> bool {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut seen = false;

        for s in cell.samples().iter() {
            if s.count() > 0 {
                seen = true;
                min_v = min_v.min(s.mean());
                max_v = max_v.max(s.mean());
            }
        }

        if include_center && cell.center_stats().count() > 0 {
            seen = true;
            min_v = min_v.min(cell.center_stats().mean());
            max_v = max_v.max(cell.center_stats().mean());
        }

        seen && min_v <= iso && max_v >= iso && (min_v < iso || max_v > iso)
    }

    /// Subdivide a cell and enqueue its children with deterministic seeds.
    fn spawn_children(&mut self, parent_index: usize) {
        let base_seed = self.params.base_seed;
        let seeds = core::array::from_fn(|i| Self::child_seed(base_seed, parent_index, i as u8));

        let children = {
            let parent = self
                .cells
                .get(parent_index)
                .expect("parent must exist")
                .clone();
            parent.subdivide_with_seeds(seeds)
        };

        if let Some(parent) = self.cells.get_mut(parent_index) {
            let cloned = core::array::from_fn(|i| Some(Box::new(children[i].clone())));
            *parent.children_mut() = cloned;
        }

        self.enqueue_children(&children);
    }

    /// Mesh a converged cell into a `MeshDelta`, cloning only the minimal data.
    fn mesh_ready_cell(&self, index: usize, mesher: &Mesher) -> MeshDelta {
        let cell = self
            .cells
            .get(index)
            .expect("cell index must exist during meshing");
        debug_assert!(
            cell.is_converged(&self.params),
            "meshing should be gated on convergence"
        );

        let positions = cell.corner_positions();
        let values = cell.corner_means();
        let grads = cell.corner_grads();
        let gradients = if grads.iter().any(|g| g.is_some()) {
            Some(grads)
        } else {
            None
        };

        mesher.mesh_cell(values, gradients, positions)
    }

    /// Combine base seed, parent index, and child id to produce a per-child seed.
    fn child_seed(base: u64, parent_index: usize, child: u8) -> u64 {
        let mix = (parent_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((child as u64) << 32)
            ^ base;
        mix ^ 0xD1B5_4A32_D192_ED03
    }

    /// Incremental mean update for gradients.
    fn accumulate_vec3(slot: &mut Option<Vec3>, count: u32, sample: Vec3) {
        let c = count as f32;
        match slot {
            Some(mean) => {
                *mean = (*mean * (c - 1.0) + sample) / c;
            }
            None => {
                *slot = Some(sample);
            }
        }
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

/// Cube→tetrahedron decomposition: six tets referencing cube corner indices (Morton order).
const TETS: [[usize; 4]; 6] = [
    [0, 5, 1, 6],
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
];

/// Tetrahedron edges as corner pairs (local tet indices 0..3).
const TET_EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];

/// Marching tetrahedra triangle table for masks 0..7 (inside = value>iso).
const TET_TRI_TABLE_POS: [&[[i8; 3]]; 8] = [
    &[],                     // 0: no vertices inside
    &[[0, 3, 2]],            // 1: 1 vertex inside
    &[[0, 1, 4]],            // 2: 1 vertex inside
    &[[1, 4, 2], [2, 4, 3]], // 3: 2 vertices inside
    &[[1, 2, 5]],
    &[[0, 3, 5], [0, 5, 1]],
    &[[0, 2, 5], [0, 5, 4]],
    &[[5, 4, 3]],
];

/// Complementary triangle table for masks 8..15 (outside mirrored); orientation preserved.
const TET_TRI_TABLE_NEG: [&[[i8; 3]]; 8] = [
    &[],
    &[[3, 4, 5]],
    &[[0, 5, 4], [0, 3, 5]],
    &[[1, 5, 0], [5, 2, 0]],
    &[[2, 3, 4], [2, 4, 1]],
    &[[1, 4, 0], [4, 3, 0]],
    &[[2, 3, 0], [3, 4, 0]],
    &[],
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a unit cube root cell with a fixed seed.
    fn root_cell() -> Cell {
        Cell::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            0,
            0xDEAD_BEEF,
        )
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

        fn phi(p: Vec3) -> f32 {
            p.length() - 2.0
        }
        fn g0(_p: Vec3) -> f32 {
            0.0
        }
        let domain: crate::SdfDomain<fn(Vec3) -> f32> = crate::SdfDomain::new(phi);
        let accel = crate::ClosestNaive;
        let solver = crate::Solver::builder(&domain, &accel).build();
        let boundary = crate::BoundaryDirichletFn::new(g0 as fn(Vec3) -> f32);
        let source = ZeroSource;

        let params = IsoParams::new(
            0.0,
            0.01,
            4,
            WalkBudget::new(1e-3, 16),
            PoissonParams::new(1),
        );
        type TD = crate::SdfDomain<fn(Vec3) -> f32>;
        type TA = crate::ClosestNaive;
        type TB = crate::BoundaryDirichletFn<fn(Vec3) -> f32>;
        type TS = ZeroSource;
        let mut sched: IsoScheduler<'_, TD, TA, TB, TS> =
            IsoScheduler::new(params, a, &solver, &boundary, &source);

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
        let seeds = core::array::from_fn(|i| i as u64 + 1);
        let children = cell.subdivide_with_seeds(seeds);
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
            assert!(
                (min.y - cell.bbox_min().y).abs() < 1e-6
                    || (min.y - mid.y).abs() < 1e-6
                    || (min.y - cell.bbox_max().y).abs() < 1e-6
            );
        }
    }

    #[test]
    fn step_updates_stats_and_requeues_when_variance_high() {
        fn phi(p: Vec3) -> f32 {
            p.length() - 2.0
        }
        fn g0(_p: Vec3) -> f32 {
            0.0
        }
        let domain: crate::SdfDomain<fn(Vec3) -> f32> = crate::SdfDomain::new(phi);
        let accel = crate::ClosestNaive;
        let solver = crate::Solver::builder(&domain, &accel).build();
        let boundary = crate::BoundaryDirichletFn::new(g0 as fn(Vec3) -> f32);
        let source = ZeroSource;

        let root = Cell::new(
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, 0.5),
            0,
            0xCAFEBABE,
        );
        // Negative tolerance guarantees variance_high=true after sampling.
        let params = IsoParams::new(
            0.0,
            -1.0,
            1,
            WalkBudget::new(1e-3, 8),
            PoissonParams::new(1),
        )
        .with_batch_samples(2)
        .with_base_seed(0xBEEFBEEF);
        let mesher = Mesher::new(params.iso_value);

        let mut sched = IsoScheduler::new(params, root, &solver, &boundary, &source);
        let _ = sched.step(&mesher);

        let cell = sched.cell(0).unwrap();
        assert!(
            cell.samples().iter().all(|s| s.count() >= 2),
            "all corners should receive samples"
        );
        assert!(
            cell.center_stats().count() >= 2,
            "center should receive samples when enabled"
        );
        // Variance high forced requeue; queue should not be empty.
        assert!(
            !sched.queue.is_empty(),
            "variance trigger should requeue when not subdividing"
        );
    }

    #[test]
    fn cell_convergence_requires_samples_and_depth_or_variance() {
        let mut cell = root_cell();
        let params = IsoParams::new(
            0.0,
            0.01,
            1,
            WalkBudget::new(1e-3, 8),
            PoissonParams::new(1),
        );

        assert!(
            !cell.is_converged(&params),
            "unsampled cell should not report convergence"
        );

        // Seed minimal samples to make the variance calculation meaningful.
        for s in cell.samples_mut().iter_mut() {
            s.push(0.5);
        }
        cell.center_stats_mut().push(0.25);
        cell.refresh_variance();
        assert!(
            cell.is_converged(&params),
            "low-variance cell with samples should converge"
        );

        // Force high variance but clamp depth to max_depth so it still converges.
        cell.set_variance(10.0);
        cell.set_depth(params.max_depth);
        assert!(
            cell.is_converged(&params),
            "depth-capped cell should count as converged even with high variance"
        );
    }

    #[test]
    fn converged_max_depth_meshes_once() {
        fn phi(p: Vec3) -> f32 {
            p.length() - 1.0
        }
        fn g0(_p: Vec3) -> f32 {
            0.0
        }

        let domain: crate::SdfDomain<fn(Vec3) -> f32> = crate::SdfDomain::new(phi);
        let accel = crate::ClosestNaive;
        let solver = crate::Solver::builder(&domain, &accel).build();
        let boundary = crate::BoundaryDirichletFn::new(g0 as fn(Vec3) -> f32);
        let source = ZeroSource;

        let root = Cell::new(
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, 0.5),
            0,
            0xBAD5EED,
        );

        // max_depth = 0 forces convergence after first sampling pass.
        let params = IsoParams::new(
            0.0,
            0.01,
            0,
            WalkBudget::new(1e-3, 8),
            PoissonParams::new(1),
        );
        let mesher = Mesher::new(params.iso_value);

        type TD = crate::SdfDomain<fn(Vec3) -> f32>;
        type TA = crate::ClosestNaive;
        type TB = crate::BoundaryDirichletFn<fn(Vec3) -> f32>;
        type TS = ZeroSource;
        let mut sched: IsoScheduler<'_, TD, TA, TB, TS> =
            IsoScheduler::new(params, root, &solver, &boundary, &source);

        let delta = sched.step(&mesher).expect("root cell should exist");
        assert!(
            delta.indices.len() <= delta.vertices.len(),
            "mesh delta should remain consistent"
        );
        assert!(
            sched.next_cell().is_none(),
            "converged cell at max depth should not be requeued"
        );
    }

    /// Constant zero source used in tests.
    #[derive(Clone, Copy)]
    struct ZeroSource;
    impl SourceTerm for ZeroSource {
        fn value(&self, _x: Vec3) -> f32 {
            0.0
        }
    }
}

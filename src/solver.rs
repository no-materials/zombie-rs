//! High-level solver façade built on top of the low-level estimators.
//!
//! [`Solver`] bundles a domain, closest-point accelerator, and optional walk
//! observers into a reusable handle. It exposes ergonomic entry points for
//! evaluating Laplace and Poisson solutions (and their gradients).

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::accel::ClosestAccel;
use crate::boundary::BoundaryDirichlet;
use crate::estimators::{
    grad_laplace_dirichlet_wos, grad_poisson_dirichlet_wos, wos_laplace_dirichlet_with_observer,
    wos_poisson_dirichlet_with_observer,
};
use crate::math::Vec3;
use crate::observer::{ObserverList, WalkObserver};
use crate::params::{GradParams, PoissonParams, WalkBudget};
use crate::rng::Rng;
use crate::source::SourceTerm;

/// Builder for [`Solver`], capturing shared configuration before freezing the solver.
pub struct SolverBuilder<'a, D, A>
where
    D: crate::domain::Domain,
    A: ClosestAccel<D>,
{
    domain: &'a D,
    accel: &'a A,
    observers: Vec<Box<dyn WalkObserver + 'a>>,
}

impl<'a, D, A> SolverBuilder<'a, D, A>
where
    D: crate::domain::Domain,
    A: ClosestAccel<D>,
{
    /// Begin constructing a solver for `domain` and `accel`.
    pub fn new(domain: &'a D, accel: &'a A) -> Self {
        Self {
            domain,
            accel,
            observers: Vec::new(),
        }
    }

    /// Register an observer that will receive walk events.
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: WalkObserver + 'a,
    {
        self.observers.push(Box::new(observer));
        self
    }

    /// Finalise the builder and produce a solver handle.
    pub fn build(self) -> Solver<'a, D, A> {
        Solver {
            domain: self.domain,
            accel: self.accel,
            observers: self.observers,
        }
    }
}

/// High-level entry point that bundles a domain, accelerator, and shared observers.
pub struct Solver<'a, D, A>
where
    D: crate::domain::Domain,
    A: ClosestAccel<D>,
{
    domain: &'a D,
    accel: &'a A,
    observers: Vec<Box<dyn WalkObserver + 'a>>,
}

impl<'a, D, A> Solver<'a, D, A>
where
    D: crate::domain::Domain,
    A: ClosestAccel<D>,
{
    /// Start constructing a solver for the given domain and accelerator.
    pub fn builder(domain: &'a D, accel: &'a A) -> SolverBuilder<'a, D, A> {
        SolverBuilder::new(domain, accel)
    }

    #[inline]
    fn observer_list(&self) -> ObserverList<'_> {
        ObserverList::new(
            self.observers
                .iter()
                .map(|obs| obs.as_ref() as &dyn WalkObserver),
        )
    }

    /// Attach an additional observer at runtime.
    pub fn add_observer<O>(&mut self, observer: O)
    where
        O: WalkObserver + 'a,
    {
        self.observers.push(Box::new(observer));
    }

    /// Solve Laplace with Dirichlet data at a query point.
    pub fn laplace_dirichlet<G>(&self, g: &G, budget: WalkBudget, rng: &mut Rng, query: Vec3) -> f32
    where
        G: BoundaryDirichlet,
    {
        let params = budget.to_wos();
        let observers = self.observer_list();
        wos_laplace_dirichlet_with_observer(
            self.domain,
            self.accel,
            g,
            params,
            rng,
            query,
            &observers,
        )
        .value
    }

    /// Solve Poisson with Dirichlet data at a query point.
    pub fn poisson_dirichlet<G, F>(
        &self,
        g: &G,
        fsrc: &F,
        walk: WalkBudget,
        poisson: PoissonParams,
        rng: &mut Rng,
        query: Vec3,
    ) -> f32
    where
        G: BoundaryDirichlet,
        F: SourceTerm,
    {
        let params = walk.to_wos();
        let observers = self.observer_list();
        wos_poisson_dirichlet_with_observer(
            self.domain,
            self.accel,
            g,
            fsrc,
            params,
            poisson,
            rng,
            query,
            &observers,
        )
        .value
    }

    /// Estimate the Laplace gradient using shared observers.
    pub fn laplace_gradient<G>(
        &self,
        g: &G,
        walk: WalkBudget,
        grad: GradParams,
        rng: &mut Rng,
        query: Vec3,
    ) -> Vec3
    where
        G: BoundaryDirichlet,
    {
        grad_laplace_dirichlet_wos(self.domain, self.accel, g, walk, grad, rng, query)
    }

    /// Estimate the Poisson gradient using shared observers.
    pub fn poisson_gradient<G, F>(
        &self,
        g: &G,
        fsrc: &F,
        walk: WalkBudget,
        poisson: PoissonParams,
        grad: GradParams,
        rng: &mut Rng,
        query: Vec3,
    ) -> Vec3
    where
        G: BoundaryDirichlet,
        F: SourceTerm,
    {
        grad_poisson_dirichlet_wos(
            self.domain,
            self.accel,
            g,
            fsrc,
            walk,
            poisson,
            grad,
            rng,
            query,
        )
    }
}

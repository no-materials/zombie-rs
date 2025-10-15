# zombie-rs

Grid-free, `no_std` Monte Carlo geometry processing kernel in Rust inspired by
Sawhney & Crane’s “Monte Carlo Geometry Processing: A Grid-Free Approach to PDE-Based Methods on Volumetric Domains”. The
crate mirrors the spirit of the original C++ reference implementation
[zombie](https://github.com/rohan-sawhney/zombie) while embracing Rust’s safety
and composability.

## Current snapshot

- Walk-on-Spheres estimators for Laplace and Poisson Dirichlet problems on mesh
  and SDF domains.
- Observer hooks for per-walk telemetry and gradient estimators that reuse the
  same interface.
- Stateless solver façade (`Solver`, `WalkBudget`) for ergonomic usage in
  `no_std` environments.

## Why Monte Carlo geometry processing?

WoS replaces volumetric grids with random walks in the continuous domain,
yielding unbiased solutions with controllable variance and trivial
parallelisation. This crate aims to provide a Rust-first, safe foundation for
research and application-level experimentation.

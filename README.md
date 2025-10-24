# zombie-rs

Grid-free, `no_std` Monte Carlo geometry processing kernel in Rust inspired by
Sawhney & Crane’s “Monte Carlo Geometry Processing: A Grid-Free Approach to PDE-Based Methods on Volumetric Domains”. The
crate mirrors the spirit of the original C++ reference implementation
[zombie](https://github.com/rohan-sawhney/zombie) while embracing Rust’s safety
and composability.

## Current snapshot

- Walk-on-Spheres estimators for Laplace and (screened) Poisson Dirichlet problems on mesh
  and SDF domains.
- Observer hooks for per-walk telemetry and gradient estimators that reuse the
  same interface.
- Stateless solver façade (`Solver`, `WalkBudget`) for ergonomic usage in
  `no_std` environments.
- ASCII PLY dump format: vertices in walk order, RGB encodes role (start = cyan,
  steps = white, boundary hit = green, max-step exit = red).

## Workspace layout

- `crates/zombie_rs/` — core library modules and examples.
- `crates/heat_probe/` — PNG heat-probe generator.
- `crates/walk_viewer/` — viewer for walk dumps.

## Quick start

```bash
cargo run -p heat_probe -- output.png 0.0 64 64
# Usage: cargo run -p heat_probe -- <output.png> [slice_z] [grid] [samples]
```
`heat_probe` generates a PNG heatmap for the requested slice.

## Why Monte Carlo geometry processing?

WoS replaces volumetric grids with random walks in the continuous domain,
yielding unbiased solutions with controllable variance and trivial
parallelisation. This workspace provides a Rust-first, safe foundation for
research and application-level experimentation.

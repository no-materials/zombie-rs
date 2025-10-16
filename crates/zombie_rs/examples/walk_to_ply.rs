use std::fs;
use std::path::PathBuf;

use zombie_rs::{
    BoundaryDirichletFn, ClosestNaive, PlyRecorder, Rng, SdfDomain, Solver, StatsObserver, Vec3,
    WalkBudget,
};

fn main() -> std::io::Result<()> {
    // Domain: unit sphere SDF
    let domain = SdfDomain::new(|p: Vec3| p.length() - 1.0);
    let accel = ClosestNaive;

    // Attach observers to capture walk statistics and dump the path as PLY.
    let stats = StatsObserver::new();
    let ply = PlyRecorder::new();
    let solver = Solver::builder(&domain, &accel)
        .with_observer(stats.clone())
        .with_observer(ply.clone())
        .build();

    // Dirichlet boundary: linear function.
    let bc = BoundaryDirichletFn::new(|p: Vec3| p.x + p.y);
    let mut rng = Rng::seed_from(42);

    // Evaluate at an interior point.
    let query = Vec3::new(0.2, 0.1, 0.0);
    let value = solver.laplace_dirichlet(&bc, WalkBudget::new(1e-4, 5_000), &mut rng, query);

    let snapshot = stats.snapshot();
    println!("Estimated u({query:?}) = {value}");
    println!(
        "Walks: {}, boundary hits: {}, max-step exits: {}, total steps: {}",
        snapshot.walks, snapshot.boundary_hits, snapshot.max_steps_hits, snapshot.total_steps
    );

    let out_path = PathBuf::from("walk.ply");
    fs::write(&out_path, ply.to_ascii())?;
    println!("Saved walk trace to {}", out_path.display());

    Ok(())
}

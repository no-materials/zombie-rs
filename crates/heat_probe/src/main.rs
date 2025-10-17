//! Heat-map probe tool for `zombie_rs` using the unit sphere domain.
//! Evaluate a 2D slice of the harmonic solution and export it as a PNG.
//!
//! Usage:
//! ```text
//! cargo run -p heat_probe -- <output.png> [slice_z] [grid] [samples_per_pixel]
//! ```
//! - `slice_z` (default `0.0`): constant z-plane in `[-1, 1]`.
//! - `grid` (default `64`): resolution of the square heatmap.
//! - `samples_per_pixel` (default `64`): number of Monte Carlo evaluations per pixel.

use std::error::Error;
use std::path::PathBuf;

use image::{ImageBuffer, Rgb};
use zombie_rs::{
    BoundaryDirichletFn, ClosestNaive, Domain, Rng, SdfDomain, Solver, Vec3, WalkBudget,
};

fn main() -> Result<(), Box<dyn Error>> {
    let config = Config::parse()?;
    let heatmap = evaluate_slice(&config)?;
    save_heatmap(&heatmap, &config.output_path, config.grid)?;
    println!(
        "Saved heat probe: {} (z = {}, grid = {}, samples = {})",
        config.output_path.display(),
        config.slice_z,
        config.grid,
        config.samples_per_pixel
    );
    Ok(())
}

/// Runtime configuration parsed from CLI arguments.
struct Config {
    output_path: PathBuf,
    slice_z: f32,
    grid: u32,
    samples_per_pixel: u32,
    epsilon: f32,
    max_steps: u32,
}

impl Config {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let mut args = std::env::args().skip(1);
        let output = args
            .next()
            .ok_or("Usage: cargo run -p heat_probe -- <output.png> [slice_z] [grid] [samples]")?;
        let slice_z = args
            .next()
            .map(|v| v.parse::<f32>().map_err(|_| "Invalid slice_z"))
            .transpose()?
            .unwrap_or(0.0);
        let grid = args
            .next()
            .map(|v| v.parse::<u32>().map_err(|_| "Invalid grid size"))
            .transpose()?
            .unwrap_or(64);
        let samples = args
            .next()
            .map(|v| v.parse::<u32>().map_err(|_| "Invalid sample count"))
            .transpose()?
            .unwrap_or(64);
        Ok(Self {
            output_path: PathBuf::from(output),
            slice_z: slice_z.clamp(-1.0, 1.0),
            grid: grid.max(1),
            samples_per_pixel: samples.max(1),
            epsilon: 1e-3,
            max_steps: 5_000,
        })
    }
}

/// Evaluate the WoS solution over a 2D grid slice.
fn evaluate_slice(config: &Config) -> Result<Vec<Option<f32>>, Box<dyn Error>> {
    let domain = SdfDomain::new(|p: Vec3| p.length() - 1.0);
    let accel = ClosestNaive;
    let solver = Solver::builder(&domain, &accel).build();
    let bc = BoundaryDirichletFn::new(|p: Vec3| p.x + p.y + p.z);
    let walk = WalkBudget::new(config.epsilon, config.max_steps);

    let grid = config.grid as usize;
    let half = 1.0f32;
    let step = (half * 2.0) / grid as f32;
    let mut values = Vec::with_capacity(grid * grid);

    for j in 0..grid {
        for i in 0..grid {
            let x = -half + (i as f32 + 0.5) * step;
            let y = -half + (j as f32 + 0.5) * step;
            let position = Vec3::new(x, y, config.slice_z);
            if !domain.is_inside(position) {
                values.push(None);
                continue;
            }
            let mut rng = Rng::seed_from(((i as u64) << 32) ^ (j as u64) ^ 0xA5A5_1234_5678);
            let mut accum = 0.0;
            for _ in 0..config.samples_per_pixel {
                let sample = solver.laplace_dirichlet(&bc, walk, &mut rng, position);
                accum += sample;
            }
            values.push(Some(accum / config.samples_per_pixel as f32));
        }
    }

    Ok(values)
}

/// Save the heatmap to disk as a PNG file.
fn save_heatmap(values: &[Option<f32>], output: &PathBuf, grid: u32) -> Result<(), Box<dyn Error>> {
    let valid: Vec<f32> = values.iter().filter_map(|v| *v).collect();
    if valid.is_empty() {
        return Err("No valid interior samples found".into());
    }
    let (min, max) = min_max(&valid);
    let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(grid, grid);

    for (idx, value) in values.iter().enumerate() {
        let x = (idx % grid as usize) as u32;
        let y = (idx / grid as usize) as u32;
        let pixel = match value {
            Some(v) => Rgb(heat_color(*v, min, max)),
            None => Rgb([0, 0, 0]),
        };
        image.put_pixel(x, grid - 1 - y, pixel);
    }

    image.save(output)?;
    Ok(())
}

fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if (max - min).abs() < f32::EPSILON {
        max = min + 1e-4;
    }
    (min, max)
}

fn heat_color(value: f32, min: f32, max: f32) -> [u8; 3] {
    let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
    // Simple blue -> cyan -> yellow -> red gradient.
    let segments = [
        (0.0, [59, 76, 192]),
        (0.25, [120, 189, 226]),
        (0.5, [197, 224, 180]),
        (0.75, [246, 170, 0]),
        (1.0, [204, 0, 0]),
    ];
    for window in segments.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if t >= t0 && t <= t1 {
            let alpha = (t - t0) / (t1 - t0);
            let blend = |a: u8, b: u8| (a as f32 + alpha * (b as f32 - a as f32)) as u8;
            return [
                blend(c0[0], c1[0]),
                blend(c0[1], c1[1]),
                blend(c0[2], c1[2]),
            ];
        }
    }
    segments.last().unwrap().1
}

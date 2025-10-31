#![cfg_attr(not(debug_assertions), warn(missing_docs))]
#![allow(clippy::needless_late_init)] // retained for clarity in GUI update logic

//! Interactive heat-probe viewer built on top of the `zombie_rs` kernel.
//!
//! This binary renders a 2D slice of the harmonic solution inside the unit
//! sphere and displays it in an `egui`+`eframe` window. A background worker
//! thread performs progressive Monte Carlo estimation, while the UI thread
//! visualises the accumulating samples and exposes live controls for the most
//! important estimator parameters.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread;

use eframe::egui::{
    self, Color32, ColorImage, Context, Slider, TextureHandle, TextureOptions, Vec2,
};
use eframe::{App, CreationContext, Frame, NativeOptions};
use rayon::prelude::*;
use zombie_rs::{
    BoundaryDirichletFn, ClosestNaive, Domain, Rng, SdfDomain, Solver, Vec3, WalkBudget,
};

/// Runtime parameters that govern how the worker evaluates the heat probe.
#[derive(Clone, Debug)]
struct ProbeParams {
    /// Constant z-slice in the unit ball.
    slice_z: f32,
    /// Number of pixels along one axis (image resolution is `grid × grid`).
    grid: u32,
    /// New Monte Carlo samples gathered per pixel in each worker pass.
    samples_per_pass: u32,
    /// Walk-on-Spheres termination threshold.
    epsilon: f32,
    /// Walk-on-Spheres maximum steps safeguard.
    max_steps: u32,
}

impl Default for ProbeParams {
    fn default() -> Self {
        Self {
            slice_z: 0.0,
            grid: 128,
            samples_per_pass: 4,
            epsilon: 1e-3,
            max_steps: 10_000,
        }
    }
}

/// Commands issued by the UI to the worker thread.
#[derive(Debug)]
enum WorkerCommand {
    /// Replace the current configuration and clear accumulated samples.
    Configure(ProbeParams),
    /// Terminate the worker loop.
    Exit,
}

/// Notifications emitted by the worker once fresh samples are ready.
#[derive(Debug)]
enum ProgressEvent {
    /// New data has been written to the accumulation buffer.
    FrameReady,
}

/// Shared accumulation buffers that the worker mutates and the UI reads.
struct ImageBuffers {
    /// Current image width in pixels.
    width: usize,
    /// Current image height in pixels.
    height: usize,
    /// Accumulated sample sums per pixel.
    accum: Vec<f32>,
    /// Number of samples contributing to each pixel.
    samples: Vec<u32>,
}

impl ImageBuffers {
    /// Create an empty buffer.
    fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            accum: Vec::new(),
            samples: Vec::new(),
        }
    }

    /// Resize the buffers and zero all accumulated data.
    fn resize_and_clear(&mut self, width: usize, height: usize) {
        let len = width * height;
        self.width = width;
        self.height = height;
        self.accum.clear();
        self.accum.resize(len, 0.0);
        self.samples.clear();
        self.samples.resize(len, 0);
    }

    /// Return an iterator-friendly snapshot of the current min/max means.
    fn min_max(&self) -> Option<(f32, f32)> {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for (sum, count) in self.accum.iter().zip(self.samples.iter()) {
            if *count == 0 {
                continue;
            }
            let mean = *sum / *count as f32;
            if mean < min {
                min = mean;
            }
            if mean > max {
                max = mean;
            }
        }
        if min.is_finite() && max.is_finite() {
            // Stabilise the colour scale if the solution is nearly constant.
            if (max - min).abs() < f32::EPSILON {
                max = min + 1e-4;
            }
            Some((min, max))
        } else {
            None
        }
    }

    /// Total number of Monte Carlo samples accumulated across the image.
    fn total_samples(&self) -> u64 {
        self.samples.iter().map(|&c| c as u64).sum()
    }
}

/// Top-level eframe application responsible for the UI and worker orchestration.
struct ProbeApp {
    /// Current UI-side configuration.
    params: ProbeParams,
    /// Shared accumulation buffers.
    buffers: Arc<Mutex<ImageBuffers>>,
    /// Channel used to push commands to the worker.
    cmd_tx: Sender<WorkerCommand>,
    /// Channel used by the worker to publish progress.
    progress_rx: Receiver<ProgressEvent>,
    /// GPU texture that mirrors the progressive heat-map.
    texture: Option<TextureHandle>,
    /// CPU staging buffer used to upload RGBA pixels to `texture`.
    upload_rgba: Vec<u8>,
    /// Last known total Monte Carlo sample count.
    latest_total_samples: u64,
    /// Tracks whether a new frame arrived and needs uploading.
    dirty: bool,
}

impl ProbeApp {
    /// Construct the app, spawn the worker thread, and kick off the first render.
    fn new(cc: &CreationContext<'_>) -> Self {
        let buffers = Arc::new(Mutex::new(ImageBuffers::new()));
        let (cmd_tx, progress_rx) = spawn_worker(buffers.clone());

        let mut app = Self {
            params: ProbeParams::default(),
            buffers,
            cmd_tx,
            progress_rx,
            texture: None,
            upload_rgba: Vec::new(),
            latest_total_samples: 0,
            dirty: false,
        };

        // Prime the worker with the default configuration.
        app.push_config();
        // Provide an initial placeholder texture so the central panel can render immediately.
        app.ensure_texture(
            &cc.egui_ctx,
            [app.params.grid as usize, app.params.grid as usize],
        );

        app
    }

    /// Send the current configuration to the worker and reset local counters.
    fn push_config(&mut self) {
        self.latest_total_samples = 0;
        let params = self.params.clone();
        self.upload_rgba.clear();
        if let Err(err) = self.cmd_tx.send(WorkerCommand::Configure(params)) {
            eprintln!("heat_probe: failed to send configuration: {err:?}");
        }
    }

    /// Rebuild `upload_rgba` and refresh the GPU texture from the shared buffers.
    fn refresh_texture(&mut self, ctx: &Context) -> bool {
        // Try to lock the shared buffers for reading.
        let guard = match self.buffers.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                // Worker currently owns the lock; try again next frame.
                return false;
            }
        };
        // If the buffers are empty, there's nothing to display.
        if guard.width == 0 || guard.height == 0 {
            return true;
        }

        // Ensure the staging RGBA buffer is the correct size.
        let len_rgba = guard.width * guard.height * 4;
        if self.upload_rgba.len() != len_rgba {
            self.upload_rgba.resize(len_rgba, 0);
        }

        // Compute the current min/max for tone-mapping.
        let range = guard.min_max();
        let (min, max) = range.unwrap_or((0.0, 1.0));
        // Update the total sample count.
        self.latest_total_samples = guard.total_samples();

        // Tone-map the floating point buffer into RGBA for the GUI.
        for (idx, rgba) in self.upload_rgba.chunks_exact_mut(4).enumerate() {
            if guard.samples[idx] == 0 {
                // Use a dark grey for unvisited pixels.
                rgba.copy_from_slice(&[12, 12, 12, 255]);
            } else {
                // Map the mean value to a heat color.
                let mean = guard.accum[idx] / guard.samples[idx] as f32;
                let rgb = heat_color(mean, min, max);
                rgba.copy_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
        }

        // Create an egui image from the RGBA buffer.
        let color_image =
            ColorImage::from_rgba_unmultiplied([guard.width, guard.height], &self.upload_rgba);
        drop(guard);

        // Update or recreate the GPU texture.
        self.ensure_texture(ctx, color_image.size);
        if let Some(texture) = self.texture.as_mut() {
            texture.set(color_image, TextureOptions::LINEAR);
        }
        true
    }

    /// Lazily create or resize the GPU texture used for display.
    fn ensure_texture(&mut self, ctx: &Context, size: [usize; 2]) {
        let needs_new = match &self.texture {
            Some(tex) => tex.size() != size,
            None => true,
        };
        if needs_new {
            let placeholder = ColorImage::new(size, Color32::BLACK);
            self.texture =
                Some(ctx.load_texture("heat_probe_texture", placeholder, TextureOptions::NEAREST));
        }
    }
}

impl App for ProbeApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // Consume progress notifications before building the UI.
        while let Ok(event) = self.progress_rx.try_recv() {
            if matches!(event, ProgressEvent::FrameReady) {
                self.dirty = true;
            }
        }
        // If new samples are available, refresh the texture.
        if self.dirty {
            if self.refresh_texture(ctx) {
                self.dirty = false;
                ctx.request_repaint();
            } else {
                // Failed to grab the buffer lock; try again soon.
                ctx.request_repaint();
            }
        }

        // Left side panel contains the controls.
        egui::SidePanel::left("controls")
            .resizable(false)
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Heat Probe");
                ui.label("Adjust the estimator and slice parameters below.");
                ui.separator();

                let mut changed = false;

                changed |= ui
                    .add(Slider::new(&mut self.params.slice_z, -1.0..=1.0).text("Slice z"))
                    .changed();
                changed |= ui
                    .add(
                        Slider::new(&mut self.params.grid, 32..=256)
                            .logarithmic(true)
                            .text("Resolution"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        Slider::new(&mut self.params.samples_per_pass, 1..=64)
                            .logarithmic(true)
                            .text("Samples/pass"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        Slider::new(&mut self.params.epsilon, 1e-5..=5e-2)
                            .logarithmic(true)
                            .text("Epsilon"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        Slider::new(&mut self.params.max_steps, 1_000..=100_000)
                            .logarithmic(true)
                            .text("Max steps"),
                    )
                    .changed();

                if ui.button("Reset accumulation").clicked() {
                    changed = true;
                }

                // If any parameter changed, push the new configuration to the worker.
                if changed {
                    self.push_config();
                }

                // Display some stats at the bottom.
                ui.separator();
                ui.label(format!("Total samples: {}", self.latest_total_samples));
                ui.label(format!("Current z: {:.3}", self.params.slice_z));
            });

        // Central panel displays the heat-map texture.
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                if let Some(texture) = &self.texture {
                    let available = ui.available_size();
                    let image_size = texture.size_vec2();
                    let scale = (available.x / image_size.x)
                        .min(available.y / image_size.y)
                        .max(0.01);
                    let draw_size = image_size * scale;
                    ui.add(egui::Image::new((texture.id(), draw_size)));
                } else {
                    ui.add_space(20.0);
                    ui.label("Waiting for samples…");
                }
            });
        });
    }
}

impl Drop for ProbeApp {
    /// Ensure the worker thread is cleanly terminated on app shutdown.
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(WorkerCommand::Exit);
    }
}

/// Spawn the Monte Carlo worker thread and return the communication channels.
fn spawn_worker(
    buffers: Arc<Mutex<ImageBuffers>>,
) -> (Sender<WorkerCommand>, Receiver<ProgressEvent>) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>();
    let (progress_tx, progress_rx) = mpsc::channel::<ProgressEvent>();

    thread::spawn(move || {
        const PIXELS_PER_BATCH: usize = 512;
        let domain = SdfDomain::new(|p: Vec3| p.length() - 1.0);
        let accel = ClosestNaive;
        let solver = Solver::builder(&domain, &accel).build();
        let bc = BoundaryDirichletFn::new(|p: Vec3| p.x + p.y + p.z);
        let mut current: Option<ProbeParams> = None;
        // Batch cursor for progressive evaluation.
        let mut cursor: usize = 0;
        // Counts how many batches have been processed (used for deterministic seeding).
        let mut pass_index: u64 = 0;

        loop {
            // Always drain the command queue first to react instantly to UI edits.
            let mut latest_config = None;
            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    WorkerCommand::Configure(params) => latest_config = Some(params),
                    WorkerCommand::Exit => return,
                }
            }
            // If there was a new configuration, apply it now.
            if let Some(params) = latest_config {
                let mut guard = match buffers.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                guard.resize_and_clear(params.grid as usize, params.grid as usize);
                drop(guard);
                current = Some(params);
                cursor = 0;
                continue;
            }

            // Ensure we have a valid configuration to work with.
            let params = match current.clone() {
                Some(cfg) => cfg,
                None => match cmd_rx.recv() {
                    Ok(WorkerCommand::Configure(cfg)) => {
                        let mut guard = match buffers.lock() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        guard.resize_and_clear(cfg.grid as usize, cfg.grid as usize);
                        drop(guard);
                        current = Some(cfg);
                        cursor = 0;
                        continue;
                    }
                    Ok(WorkerCommand::Exit) | Err(_) => return,
                },
            };

            let walk = WalkBudget::new(params.epsilon, params.max_steps);
            let grid = params.grid as usize;
            let total_pixels = grid.saturating_mul(grid);
            if total_pixels == 0 {
                thread::yield_now();
                continue;
            }

            // Precompute constants for mapping pixel indices to positions.
            let half = 1.0f32;
            let step = (half * 2.0) / grid as f32;

            // Determine the next batch of pixels to process.
            let batch = PIXELS_PER_BATCH.min(total_pixels);
            let indices: Vec<usize> = (0..batch).map(|n| (cursor + n) % total_pixels).collect();
            cursor = (cursor + batch) % total_pixels;

            // Capture the current pass index for deterministic seeding.
            let pass_id = pass_index;
            pass_index = pass_index.wrapping_add(1);

            let updates: Vec<(usize, f32, u32)> = indices
                .par_iter()
                .filter_map(|&idx| {
                    let i = idx % grid;
                    let j = idx / grid;
                    let x = -half + (i as f32 + 0.5) * step;
                    let y = -half + (j as f32 + 0.5) * step;
                    let position = Vec3::new(x, y, params.slice_z);

                    // Skip pixels outside the domain.
                    if !domain.is_inside(position) {
                        return None;
                    }

                    // Derive a deterministic per-pixel seed.
                    let mut seed =
                        splitmix64((idx as u64) ^ pass_id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    if let Some(tid) = rayon::current_thread_index() {
                        seed ^= (tid as u64).rotate_left(17);
                    }
                    let mut local_rng = Rng::seed_from(seed);

                    // Accumulate samples for this pixel.
                    let mut sum = 0.0f32;
                    for _ in 0..params.samples_per_pass {
                        let sample = solver.laplace_dirichlet(&bc, walk, &mut local_rng, position);
                        sum += sample;
                    }
                    Some((idx, sum, params.samples_per_pass))
                })
                .collect();

            if !updates.is_empty() {
                let mut guard = match buffers.lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                for (idx, sum, count) in updates {
                    guard.accum[idx] += sum;
                    guard.samples[idx] += count;
                }
            }

            // Notify the UI that new data is available.
            let _ = progress_tx.send(ProgressEvent::FrameReady);
            thread::yield_now();
        }
    });

    (cmd_tx, progress_rx)
}

/// SplitMix64 PRNG mixer used to derive deterministic seeds for parallel workers.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Traditional blue→red heat-map used to visualise harmonic values.
fn heat_color(value: f32, min: f32, max: f32) -> [u8; 3] {
    let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
    // 5-stop gradient from the original CLI tool.
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

fn main() -> eframe::Result<()> {
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(Vec2::new(960.0, 720.0))
            .with_min_inner_size(Vec2::new(640.0, 480.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Heat Probe Viewer",
        options,
        Box::new(|cc| Box::new(ProbeApp::new(cc))),
    )
}

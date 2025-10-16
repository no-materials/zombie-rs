//! Minimal three-d viewer for `zombie_rs` walk dumps.
//!
//! Launch with `cargo run -p walk_viewer -- <path/to/walk.ply>`.

use std::fs;
use std::path::PathBuf;

use three_d::*;

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("Please provide a PLY file path");
    let contents = fs::read_to_string(&path).expect("Failed to read path");
    let points = parse_ascii_ply(&contents).expect("Failed to parse PLY");

    let window = Window::new(WindowSettings {
        title: "Walk Viewer".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(0.125, -0.25, -0.5),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.01,
        100.0,
    );
    let mut control = OrbitControl::new(camera.target(), 0.1, 3.0);

    let point_cloud = build_point_cloud(points);

    let mut point_mesh = CpuMesh::sphere(4);
    point_mesh.transform(Mat4::from_scale(0.01)).unwrap();
    let instanced = InstancedMesh::new(&context, &point_cloud.into(), &point_mesh);

    let mut point_cloud = Gm::new(
        instanced,
        PhysicalMaterial::new_transparent(
            &context,
            &CpuMaterial {
                albedo: Srgba {
                    r: 255,
                    g: 0,
                    b: 0,
                    a: 200,
                },
                ..Default::default()
            },
        ),
    );
    let c = -point_cloud.aabb().center();
    point_cloud.set_transformation(Mat4::from_translation(c));

    // Main loop.
    window.render_loop(move |mut frame_input| {
        let mut redraw = frame_input.first_frame;
        redraw |= camera.set_viewport(frame_input.viewport);
        redraw |= control.handle_events(&mut camera, &mut frame_input.events);

        if redraw {
            frame_input
                .screen()
                .clear(ClearState::color_and_depth(1.0, 1.0, 1.0, 1.0, 1.0))
                .render(
                    &camera,
                    point_cloud
                        .into_iter()
                        .chain(&Axes::new(&context, 0.01, 0.1)),
                    &[],
                );
        }

        FrameOutput {
            swap_buffers: redraw,
            ..Default::default()
        }
    });
}

#[derive(Clone, Debug)]
struct WalkPoint {
    position: Vec3,
    color: Srgba,
}

fn parse_ascii_ply(contents: &str) -> Result<Vec<WalkPoint>, String> {
    let mut lines = contents.lines();
    let header = lines.next().ok_or_else(|| "Empty file".to_string())?;
    if header.trim() != "ply" {
        return Err("Expected `ply` header".into());
    }
    let format = lines
        .next()
        .ok_or_else(|| "Missing format line".to_string())?;
    if format.trim() != "format ascii 1.0" {
        return Err("Only `format ascii 1.0` is supported".into());
    }

    let mut vertex_count: Option<usize> = None;
    for line in &mut lines {
        let trimmed = line.trim();
        if trimmed.starts_with("element vertex") {
            let parts: Vec<_> = trimmed.split_whitespace().collect();
            if let Some(count_str) = parts.get(2) {
                vertex_count = Some(
                    count_str
                        .parse::<usize>()
                        .map_err(|_| "Invalid vertex count".to_string())?,
                );
            }
        } else if trimmed == "end_header" {
            break;
        }
    }

    let expected = vertex_count.ok_or_else(|| "Missing vertex count".to_string())?;
    let mut points = Vec::with_capacity(expected);

    for (index, line) in lines.take(expected).enumerate() {
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.len() < 6 {
            return Err(format!("Vertex line {index} malformed"));
        }
        let parse_f32 = |s: &str| -> Result<f32, String> {
            s.parse::<f32>().map_err(|_| format!("Invalid float `{s}`"))
        };
        let parse_color = |s: &str| -> Result<u8, String> {
            s.parse::<u8>()
                .map_err(|_| format!("Invalid color component `{s}`"))
        };
        let position = vec3(
            parse_f32(parts[0])?,
            parse_f32(parts[1])?,
            parse_f32(parts[2])?,
        );
        let color = Srgba::new(
            parse_color(parts[3])?,
            parse_color(parts[4])?,
            parse_color(parts[5])?,
            255,
        );
        points.push(WalkPoint { position, color });
    }

    Ok(points)
}

fn build_point_cloud(points: Vec<WalkPoint>) -> PointCloud {
    let pc = PointCloud {
        positions: Positions::F32(points.iter().map(|p| p.position).collect()),
        colors: Some(points.iter().map(|p| p.color).collect()),
    };
    pc
}

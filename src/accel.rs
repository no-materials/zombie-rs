//! Closest-point accelerators such as BVHs.

use alloc::vec::Vec;
use core::cmp::Ordering;
use libm::sqrtf;

use crate::domain::{Closest, Domain, PolygonSoupDomain};
use crate::math::{closest_point_on_triangle, Aabb, Vec3};

/// A single BVH node.
/// We store a binary tree with nodes laid out in a flat array for cache locality.
#[derive(Copy, Clone, Debug)]
struct BvhNode {
    aabb: Aabb,
    // If leaf: range over primitive indices [start, count); if inner: left/right child indices
    start: u32,
    count: u32,
    left: u32,
    right: u32,
    is_leaf: bool,
}

/// BVH over triangle indices for closest-point acceleration.
///
/// Build: longest-axis median split. Query: branch-and-bound over leaf triangles.
pub struct Bvh {
    nodes: Vec<BvhNode>,
    prims: Vec<u32>, // permutation of triangle indices
}

impl Bvh {
    /// Build a BVH for a polygon soup domain.
    pub fn build(mesh: &PolygonSoupDomain) -> Self {
        // Primitive bounds: triangle AABBs
        let mut prims: Vec<u32> = (0..mesh.tris.len() as u32).collect();
        let mut nodes: Vec<BvhNode> = Vec::new();
        nodes.reserve(mesh.tris.len() * 2);

        fn tri_aabb(mesh: &PolygonSoupDomain, i: usize) -> Aabb {
            let (a, b, c) = mesh.tri(i);
            Aabb::from_points(a, b, c)
        }

        fn centroid(mesh: &PolygonSoupDomain, i: usize) -> Vec3 {
            let (a, b, c) = mesh.tri(i);
            (a + b + c) / 3.0
        }

        fn build_rec(mesh: &PolygonSoupDomain, nodes: &mut Vec<BvhNode>, prims: &mut [u32]) -> u32 {
            // Compute node bounds
            let mut node_aabb = Aabb::empty();
            for &pi in prims.iter() {
                node_aabb = node_aabb.union(tri_aabb(mesh, pi as usize));
            }

            let node_index = nodes.len() as u32;
            // Leaf threshold: small ranges
            if prims.len() <= 8 {
                nodes.push(BvhNode {
                    aabb: node_aabb,
                    start: 0, // filled later
                    count: prims.len() as u32,
                    left: 0,
                    right: 0,
                    is_leaf: true,
                });
                return node_index;
            }

            // Split along longest axis by median on primitive centroids
            let axis = node_aabb.longest_axis();
            prims.sort_unstable_by(|&a, &b| {
                let ca = centroid(mesh, a as usize);
                let cb = centroid(mesh, b as usize);
                let oa = match axis {
                    0 => ca.x,
                    1 => ca.y,
                    _ => ca.z,
                };
                let ob = match axis {
                    0 => cb.x,
                    1 => cb.y,
                    _ => cb.z,
                };
                oa.partial_cmp(&ob).unwrap_or(Ordering::Equal)
            });
            let mid = prims.len() / 2;
            let (left_slice, right_slice) = prims.split_at_mut(mid);

            // Placeholder, to be patched after recursion:
            let dummy = BvhNode {
                aabb: node_aabb,
                start: 0,
                count: 0,
                left: 0,
                right: 0,
                is_leaf: false,
            };
            nodes.push(dummy);

            let left_index = build_rec(mesh, nodes, left_slice);
            let right_index = build_rec(mesh, nodes, right_slice);

            // Patch internal node
            let node = &mut nodes[node_index as usize];
            node.left = left_index;
            node.right = right_index;

            node_index
        }

        // Build hierarchy
        let root = build_rec(mesh, &mut nodes, &mut prims[..]);

        // Flatten leaf ranges: assign contiguous ranges in `prims`
        // We perform a DFS to remap leaf prims into contiguous order and set start/count.
        let mut remap: Vec<u32> = Vec::with_capacity(prims.len());
        fn assign_ranges(
            mesh: &PolygonSoupDomain,
            nodes: &mut [BvhNode],
            prims: &mut [u32],
            remap: &mut Vec<u32>,
            idx: u32,
        ) {
            let node = nodes[idx as usize];
            if node.is_leaf {
                let start = remap.len() as u32;
                remap.extend_from_slice(prims);
                let count = (remap.len() as u32) - start;
                let mut n = node;
                n.start = start;
                n.count = count;
                nodes[idx as usize] = n;
            } else {
                // Split prims same way as build to maintain correspondence
                let axis = node.aabb.longest_axis();
                prims.sort_unstable_by(|&a, &b| {
                    let ca = {
                        let (pa, pb, pc) = mesh.tri(a as usize);
                        (pa + pb + pc) / 3.0
                    };
                    let cb = {
                        let (pa, pb, pc) = mesh.tri(b as usize);
                        (pa + pb + pc) / 3.0
                    };
                    let oa = match axis {
                        0 => ca.x,
                        1 => ca.y,
                        _ => ca.z,
                    };
                    let ob = match axis {
                        0 => cb.x,
                        1 => cb.y,
                        _ => cb.z,
                    };
                    oa.partial_cmp(&ob).unwrap_or(Ordering::Equal)
                });
                let mid = prims.len() / 2;
                let (left_slice, right_slice) = prims.split_at_mut(mid);
                assign_ranges(mesh, nodes, left_slice, remap, nodes[idx as usize].left);
                assign_ranges(mesh, nodes, right_slice, remap, nodes[idx as usize].right);
            }
        }
        assign_ranges(mesh, &mut nodes[..], &mut prims[..], &mut remap, root);

        Self {
            nodes,
            prims: remap,
        }
    }

    /// Closest point on the boundary for polygon soup.
    ///
    /// Returns `(Closest, best_tri_index)`.
    pub fn closest_point(&self, mesh: &PolygonSoupDomain, x: Vec3) -> (Closest, u32) {
        // Stack-based traversal (no recursion)
        let mut stack: [u32; 64] = [0; 64];
        let mut sp = 0usize;
        if self.nodes.is_empty() {
            return (Closest::empty(), u32::MAX);
        }
        stack[sp] = 0;
        sp += 1;

        let mut best_d2 = f32::INFINITY;
        let mut best_point = Vec3::new(0.0, 0.0, 0.0);
        let mut best_nrm = Vec3::new(0.0, 1.0, 0.0);
        let mut best_tri = u32::MAX;

        while sp > 0 {
            sp -= 1;
            let idx = stack[sp];
            let node = self.nodes[idx as usize];
            // Prune by AABB distance
            if node.aabb.distance_sq(x) >= best_d2 {
                continue;
            }
            if node.is_leaf {
                let start = node.start as usize;
                let end = start + node.count as usize;
                for i in start..end {
                    let tri_idx = self.prims[i] as usize;
                    let (a, b, c) = mesh.tri(tri_idx);
                    let q = closest_point_on_triangle(x, a, b, c);
                    let d2 = (x - q).length_sq();
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best_point = q;
                        best_nrm = mesh.tri_normals[tri_idx]; // facet normal (approx.)
                        best_tri = tri_idx as u32;
                    }
                }
            } else {
                // Visit nearer child first (AABB heuristic)
                let l = node.left;
                let r = node.right;
                let dl = self.nodes[l as usize].aabb.distance_sq(x);
                let dr = self.nodes[r as usize].aabb.distance_sq(x);
                if dl < dr {
                    stack[sp] = r;
                    sp += 1;
                    stack[sp] = l;
                    sp += 1;
                } else {
                    stack[sp] = l;
                    sp += 1;
                    stack[sp] = r;
                    sp += 1;
                }
            }
        }

        let dist = sqrtf(best_d2);
        (
            Closest {
                point: best_point,
                normal: best_nrm,
                distance: dist,
            },
            best_tri,
        )
    }
}

/// Allows plugging in different acceleration structures (e.g., BVH, spatial hashing).
pub trait ClosestAccel<D: Domain>: Send + Sync {
    fn closest(&self, domain: &D, x: Vec3) -> Closest;
}

/// A no-acceleration adapter (O(n) scan) for any `Domain`.
pub struct ClosestNaive;

impl<D: Domain> ClosestAccel<D> for ClosestNaive {
    #[inline]
    fn closest(&self, domain: &D, x: Vec3) -> Closest {
        domain.closest(x)
    }
}

/// BVH accelerator for `PolygonSoupDomain`.
pub struct BvhAccel<'a> {
    bvh: &'a Bvh,
    mesh: &'a PolygonSoupDomain,
}

impl<'a> BvhAccel<'a> {
    pub fn new(bvh: &'a Bvh, mesh: &'a PolygonSoupDomain) -> Self {
        Self { bvh, mesh }
    }
}

impl<'a> ClosestAccel<PolygonSoupDomain> for BvhAccel<'a> {
    #[inline]
    fn closest(&self, _domain: &PolygonSoupDomain, x: Vec3) -> Closest {
        self.bvh.closest_point(self.mesh, x).0
    }
}

//! Conservative CPU visibility for Vulkan clip space (x/y in [-w,w], z in [0,w]).
use glam::{Mat4, Vec3};
use govno_core::BoxError;
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}
impl Aabb {
    pub fn transformed(self, matrix: Mat4) -> Option<Self> {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for x in [self.min.x, self.max.x] {
            for y in [self.min.y, self.max.y] {
                for z in [self.min.z, self.max.z] {
                    let p = matrix.transform_point3(Vec3::new(x, y, z));
                    if !p.is_finite() {
                        return None;
                    }
                    min = min.min(p);
                    max = max.max(p);
                }
            }
        }
        Some(Self { min, max })
    }
}
#[derive(Debug, Clone)]
pub struct Frustum {
    planes: [[f64; 4]; 6],
}
impl Frustum {
    pub fn from_view_projection(matrix: Mat4) -> Result<Self, BoxError> {
        if !matrix.is_finite() {
            return Err("frustum matrix must be finite".into());
        }
        let rows = matrix
            .transpose()
            .to_cols_array_2d()
            .map(|row| row.map(f64::from));
        let combine =
            |a: usize, b: usize, sign: f64| std::array::from_fn(|i| rows[a][i] + sign * rows[b][i]);
        Ok(Self {
            planes: [
                combine(3, 0, 1.0),
                combine(3, 0, -1.0),
                combine(3, 1, 1.0),
                combine(3, 1, -1.0),
                rows[2],
                combine(3, 2, -1.0),
            ],
        })
    }
    pub fn intersects(&self, bounds: Aabb) -> bool {
        self.planes.iter().all(|p| {
            let vertex = [
                if p[0] >= 0.0 {
                    bounds.max.x
                } else {
                    bounds.min.x
                },
                if p[1] >= 0.0 {
                    bounds.max.y
                } else {
                    bounds.min.y
                },
                if p[2] >= 0.0 {
                    bounds.max.z
                } else {
                    bounds.min.z
                },
            ];
            let terms = [
                p[0] * f64::from(vertex[0]),
                p[1] * f64::from(vertex[1]),
                p[2] * f64::from(vertex[2]),
                p[3],
            ];
            // Relative tolerance conservatively retains boxes touching a plane. A constant
            // positive plane naturally represents the infinite reversed-Z far boundary.
            terms.iter().sum::<f64>() >= -1e-7 * terms.iter().map(|v| v.abs()).sum::<f64>().max(1.0)
        })
    }
}
impl crate::MeshData {
    pub fn bounds(&self) -> Result<Aabb, BoxError> {
        self.validate()?;
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for vertex in &self.vertices {
            let p = Vec3::from_array(vertex.position);
            min = min.min(p);
            max = max.max(p);
        }
        Ok(Aabb { min, max })
    }
}

"""Geometry operations in SI units. Face-corner attributes survive topology edits."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import trimesh
import mapbox_earcut
from scipy.spatial.transform import Rotation


def rotation(q):
    q = np.asarray(q, float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def quaternion(r):
    return Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]


def valid_inertia(mass, center, tensor):
    tensor = np.asarray(tensor, float).reshape(3, 3)
    if not np.isfinite([mass, *center, *tensor.flat]).all() or mass <= 0:
        raise ValueError("质量必须为正，惯性参数必须有限")
    if not np.allclose(tensor, tensor.T, atol=1e-12):
        raise ValueError("惯性张量必须对称")
    eig = np.linalg.eigvalsh(tensor)
    if eig[0] <= 0 or eig[-1] > eig[:2].sum() + 1e-10 * eig[-1]:
        raise ValueError("惯性不满足正定性或主惯量三角不等式")
    return tensor


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None  # face, corner, uv
    normals: np.ndarray | None = None
    materials: np.ndarray | None = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int32).reshape(-1, 3)
        if not np.isfinite(self.vertices).all() or not len(self.faces):
            raise ValueError("网格为空或坐标不是有限数")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("三角面顶点索引越界")

    def trimesh(self):
        # Weld positions only for physical topology, never discard seam attributes.
        result = trimesh.Trimesh(self.vertices.copy(), self.faces.copy(), process=False)
        result.merge_vertices()
        return result

    def json(self):
        return {
            "vertices": self.vertices.tolist(),
            "faces": self.faces.tolist(),
            "uv": None if self.uv is None else self.uv.tolist(),
            "normals": None if self.normals is None else self.normals.tolist(),
            "materials": None if self.materials is None else self.materials.tolist(),
        }

    @classmethod
    def from_json(cls, value):
        return cls(
            **{k: None if v is None else np.asarray(v) for k, v in value.items()}
        )

    def subset(self, indices):
        indices = np.asarray(indices, dtype=int)
        if not len(indices):
            raise ValueError("拆分两侧都必须包含三角面")
        faces = self.faces[indices]
        used, inverse = np.unique(faces, return_inverse=True)
        return Mesh(
            self.vertices[used],
            inverse.reshape(-1, 3),
            None if self.uv is None else self.uv[indices],
            None if self.normals is None else self.normals[indices],
            None if self.materials is None else self.materials[indices],
        )

    def split(self, indices):
        chosen = np.unique(np.asarray(indices, dtype=int))
        if not len(chosen) or chosen.min() < 0 or chosen.max() >= len(self.faces):
            raise ValueError("请选择有效三角面")
        return self.subset(chosen), self.subset(
            np.setdiff1d(np.arange(len(self.faces)), chosen)
        )

    def boundaries(self):
        mesh = self.trimesh()
        edges = np.sort(mesh.edges, axis=1)
        _, inverse, counts = np.unique(
            edges, axis=0, return_inverse=True, return_counts=True
        )
        directed = mesh.edges[counts[inverse] == 1]
        if not len(directed):
            return [], mesh
        outgoing = {}
        incoming = {}
        for a, b in directed:
            outgoing.setdefault(int(a), []).append(int(b))
            incoming.setdefault(int(b), []).append(int(a))
        if any(
            len(outgoing.get(v, [])) != 1 or len(incoming.get(v, [])) != 1
            for v in set(outgoing) | set(incoming)
        ):
            raise ValueError("边界不是简单闭环，需修复非流形或法向")
        loops = []
        remaining = set(outgoing)
        while remaining:
            start = min(remaining)
            loop = [start]
            remaining.remove(start)
            nxt = outgoing[start][0]
            while nxt != start:
                if nxt not in remaining:
                    raise ValueError("边界环交叉")
                loop.append(nxt)
                remaining.remove(nxt)
                nxt = outgoing[nxt][0]
            loops.append(loop)
        return loops, mesh

    def diagnostics(self):
        mesh = self.trimesh()
        _, count = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
        scale = max(float(np.ptp(mesh.vertices, axis=0).max()), 1e-12)
        result = {
            "faces": len(mesh.faces),
            "vertices": len(mesh.vertices),
            "boundary_edges": int(np.sum(count == 1)),
            "nonmanifold_edges": int(np.sum(count > 2)),
            "degenerate_faces": int(np.sum(mesh.area_faces <= scale * scale * 1e-14)),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "watertight": bool(mesh.is_watertight),
            "volume": float(
                np.einsum(
                    "ij,ij->i",
                    mesh.triangles[:, 0],
                    np.cross(mesh.triangles[:, 1], mesh.triangles[:, 2]),
                ).sum()
                / 6
            ),
            "self_intersections": intersections(mesh),
        }
        result["valid_solid"] = bool(
            result["watertight"]
            and result["winding_consistent"]
            and not result["nonmanifold_edges"]
            and not result["degenerate_faces"]
            and not result["self_intersections"]
            and result["volume"] > 0
        )
        try:
            loops, boundary_mesh = self.boundaries()
            result["boundaries"] = [
                boundary_mesh.vertices[loop].tolist() for loop in loops
            ]
        except ValueError as exc:
            result["boundaries"] = []
            result["boundary_error"] = str(exc)
        return result

    def cap(self):
        loops, welded = self.boundaries()
        if not loops:
            raise ValueError("没有需要补洞的边界")
        # Preserve all original corners. Add new corners for planar caps with local UV.
        vertices, faces = self.vertices.tolist(), self.faces.tolist()
        uv = (np.zeros((len(faces), 3, 2)) if self.uv is None else self.uv).tolist()
        normals = (
            trimesh.Trimesh(self.vertices, self.faces, process=False).vertex_normals[
                self.faces
            ]
            if self.normals is None
            else self.normals
        ).tolist()
        materials = (
            np.zeros(len(faces), int) if self.materials is None else self.materials
        ).tolist()
        cap_material = max(materials, default=0) + 1
        for loop in loops:
            points = welded.vertices[loop][::-1]
            center = points.mean(axis=0)
            _, _, vt = np.linalg.svd(points - center)
            extent = max(float(np.ptp(points, axis=0).max()), 1e-12)
            if np.max(np.abs((points - center) @ vt[2])) > extent * 1e-6:
                raise ValueError("非平面边界不能可靠自动补洞；请手动修复实体")
            xy = (points - center) @ vt[:2].T
            indices = mapbox_earcut.triangulate_float64(
                xy, np.array([len(xy)], np.uint32)
            ).reshape(-1, 3)
            if len(indices) != len(points) - 2:
                raise ValueError("边界三角化失败")
            # earcut orientation is independent of input order; match reversed boundary.
            target = np.cross(points[1] - points[0], points[2] - points[1])
            if np.linalg.norm(target) < 1e-12 * extent * extent:
                target = np.sum(
                    np.cross(points - center, np.roll(points - center, -1, axis=0)),
                    axis=0,
                )
            a, b, c = points[indices[0]]
            if np.dot(np.cross(b - a, c - a), target) < 0:
                indices = indices[:, ::-1]
            offset = len(vertices)
            vertices.extend(points.tolist())
            faces.extend((indices + offset).tolist())
            uv.extend(xy[indices].tolist())
            n = target / np.linalg.norm(target)
            normals.extend(np.broadcast_to(n, (len(indices), 3, 3)).tolist())
            materials.extend([cap_material] * len(indices))
        result = Mesh(
            vertices, faces, np.asarray(uv), np.asarray(normals), np.asarray(materials)
        )
        if not result.diagnostics()["valid_solid"]:
            raise ValueError("补洞结果未通过实体检查，未修改原网格")
        return result

    def mass_properties(self, density=1000.0, mass=None):
        diagnostics = self.diagnostics()
        if not diagnostics["valid_solid"]:
            raise ValueError("网格不是有效封闭实体，请补洞或使用手动惯性")
        mesh = self.trimesh()
        if mass is not None:
            density = float(mass) / mesh.volume
        if not np.isfinite(density) or density <= 0:
            raise ValueError("密度或实测质量必须大于零")
        mesh.density = density
        p = mesh.mass_properties
        valid_inertia(p.mass, p.center_mass, p.inertia)
        return {
            "mass": float(p.mass),
            "center": p.center_mass.tolist(),
            "tensor": p.inertia.tolist(),
            "volume": float(p.volume),
            "density": float(density),
        }


def intersections(mesh):
    """AABB sweep + triangle SAT, ignoring topological neighbors; coplanar overlap tested in 2D."""
    triangles = mesh.triangles
    low, high = triangles.min(axis=1), triangles.max(axis=1)
    order = np.argsort(low[:, 0])
    scale = max(float(np.ptp(mesh.vertices, axis=0).max()), 1e-12)
    eps = scale * 1e-10
    found = []
    for oi, i in enumerate(order):
        candidates = order[oi + 1 :]
        candidates = candidates[low[candidates, 0] <= high[i, 0] + eps]
        candidates = candidates[
            np.all(high[candidates] >= low[i] - eps, axis=1)
            & np.all(low[candidates] <= high[i] + eps, axis=1)
        ]
        for j in candidates:
            if np.intersect1d(mesh.faces[i], mesh.faces[j]).size:
                continue
            a, b = triangles[i], triangles[j]
            ea, eb = np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b
            na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
            axes = [na, nb] + [np.cross(x, y) for x in ea for y in eb]
            if np.linalg.norm(np.cross(na, nb)) <= 1e-10 * np.linalg.norm(
                na
            ) * np.linalg.norm(nb):
                axes += [np.cross(na, e) for e in [*ea, *eb]]
            separated = False
            for axis in axes:
                norm = np.linalg.norm(axis)
                if norm < 1e-25:
                    continue
                axis = axis / norm
                pa, pb = a @ axis, b @ axis
                if pa.max() < pb.min() - eps or pb.max() < pa.min() - eps:
                    separated = True
                    break
            if not separated:
                found.append([int(i), int(j)])
                if len(found) >= 100:
                    return found  # diagnostics explicitly bounded; any hit invalidates the solid
    return found


def combine(properties):
    total = sum(p["mass"] for p in properties)
    if total <= 0:
        raise ValueError("请选择至少一个有效实体")
    center = sum(p["mass"] * np.asarray(p["center"]) for p in properties) / total
    tensor = np.zeros((3, 3))
    for p in properties:
        delta = np.asarray(p["center"]) - center
        tensor += np.asarray(p["tensor"]) + p["mass"] * (
            np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)
        )
    valid_inertia(total, center, tensor)
    return {"mass": total, "center": center.tolist(), "tensor": tensor.tolist()}


def primitive_properties(kind, size, density):
    """Analytic solid inertia about the primitive origin (MuJoCo size conventions)."""
    a, b, c = np.asarray(size, float)
    if not np.isfinite([a, b, c, density]).all() or density <= 0:
        raise ValueError("尺寸和密度必须有效")
    if kind == 6:  # box, half extents
        volume = 8 * a * b * c
        mass = density * volume
        diagonal = mass / 3 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    elif kind == 2:  # sphere
        volume = 4 * np.pi * a**3 / 3
        mass = density * volume
        diagonal = np.full(3, 2 * mass * a * a / 5)
    elif kind == 4:  # ellipsoid
        volume = 4 * np.pi * a * b * c / 3
        mass = density * volume
        diagonal = mass / 5 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    elif kind == 5:  # cylinder along local z; b is half length
        volume = np.pi * a * a * 2 * b
        mass = density * volume
        diagonal = np.array(
            [mass * (3 * a * a + 4 * b * b) / 12] * 2 + [mass * a * a / 2]
        )
    elif kind == 3:  # cylinder with two solid hemispheres
        mc = density * np.pi * a * a * 2 * b
        ms = density * 4 * np.pi * a**3 / 3
        mass = mc + ms
        volume = mass / density
        transverse = mc * (3 * a * a + 4 * b * b) / 12 + ms * (
            0.4 * a * a + 0.75 * b * a + b * b
        )
        diagonal = np.array([transverse, transverse, mc * a * a / 2 + 0.4 * ms * a * a])
    else:
        raise ValueError("该几何不表示支持的实体；请使用手动惯性")
    tensor = np.diag(diagonal)
    valid_inertia(mass, [0, 0, 0], tensor)
    return {
        "mass": float(mass),
        "center": [0.0, 0.0, 0.0],
        "tensor": tensor.tolist(),
        "volume": float(volume),
        "density": float(density),
    }

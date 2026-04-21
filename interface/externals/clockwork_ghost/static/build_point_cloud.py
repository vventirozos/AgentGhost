#!/usr/bin/env python3
"""
Sample an OBJ mesh surface uniformly and emit a JSON point cloud
suitable for THREE.Points rendering. Weights samples by triangle area
so points distribute evenly across the surface.
"""
import json
import sys
import random
from pathlib import Path

SRC = Path(__file__).parent / "head.obj"
DST = Path(__file__).parent / "head_points.json"
TARGET_POINTS = 22000

def load_obj(path):
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                # "f v/vt/vn v/vt/vn ..." — take first index of each token, OBJ is 1-indexed
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                # triangulate fan if polygon has >3 verts
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    return verts, faces

def tri_area(a, b, c):
    # |(b-a) x (c-a)| / 2
    ux, uy, uz = b[0]-a[0], b[1]-a[1], b[2]-a[2]
    vx, vy, vz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
    cx = uy*vz - uz*vy
    cy = uz*vx - ux*vz
    cz = ux*vy - uy*vx
    return 0.5 * (cx*cx + cy*cy + cz*cz) ** 0.5

def sample_triangle(a, b, c):
    r1 = random.random()
    r2 = random.random()
    s1 = r1 ** 0.5
    u = 1.0 - s1
    v = r2 * s1
    w = 1.0 - u - v
    return (
        u*a[0] + v*b[0] + w*c[0],
        u*a[1] + v*b[1] + w*c[1],
        u*a[2] + v*b[2] + w*c[2],
    )

def main():
    verts, faces = load_obj(SRC)
    print(f"loaded {len(verts)} verts, {len(faces)} triangles")

    areas = [tri_area(verts[i], verts[j], verts[k]) for i, j, k in faces]
    total_area = sum(areas)
    print(f"total surface area: {total_area:.2f}")

    # allocate points to each triangle proportional to area
    points = []
    leftover = 0.0
    for (i, j, k), area in zip(faces, areas):
        exact = TARGET_POINTS * area / total_area + leftover
        n = int(exact)
        leftover = exact - n
        a, b, c = verts[i], verts[j], verts[k]
        for _ in range(n):
            points.append(sample_triangle(a, b, c))

    # also include original vertices to keep sharp features crisp
    points.extend(verts)
    print(f"generated {len(points)} points")

    # flatten + center on origin
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    scale = 20.0 / size  # normalize to ~20 unit bbox

    flat = []
    for p in points:
        flat.append(round((p[0] - cx) * scale, 4))
        flat.append(round((p[1] - cy) * scale, 4))
        flat.append(round((p[2] - cz) * scale, 4))

    DST.write_text(json.dumps({"positions": flat, "count": len(points)}))
    print(f"wrote {DST} ({DST.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()

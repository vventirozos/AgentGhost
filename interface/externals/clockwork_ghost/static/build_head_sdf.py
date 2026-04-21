#!/usr/bin/env python3
"""
Generate a head-shaped point cloud procedurally from a signed distance field.

No external mesh required — we compose the head from analytic primitives
(skull, jaw, cheeks, nose, brow, eye sockets, neck, ears) using smooth min/max,
then sample points directly on the zero isosurface by projecting candidates
along the SDF gradient.

Output: head_points.json  (same schema face3d_points.html already reads)
"""
import json
import math
import random
from pathlib import Path

import numpy as np

DST = Path(__file__).parent / "head_points.json"
TARGET_POINTS = 28000

# ---- SDF primitives ----

def sd_ellipsoid(p, r):
    # inexact but smooth ellipsoid distance (good enough near surface)
    k0 = np.linalg.norm(p / r, axis=-1)
    k1 = np.linalg.norm(p / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)

def sd_sphere(p, r):
    return np.linalg.norm(p, axis=-1) - r

def sd_capsule(p, a, b, r):
    pa = p - a
    ba = b - a
    h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    return np.linalg.norm(pa - ba * h[..., None], axis=-1) - r

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1 - h) + a * h) - k * h * (1 - h)

def smax(a, b, k):
    return -smin(-a, -b, k)

# ---- The head ----
#
# Coordinate system: +Y up, +Z forward (face points +Z), +X to the viewer's right.
# Units are roughly "one head = 2 tall".

def head_sdf(p):
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    # ---- skull (tall ellipsoid, a little deeper than wide) ----
    skull = sd_ellipsoid(
        np.stack([x, (y - 0.15), z * 0.95], axis=-1),
        np.array([0.85, 1.05, 0.95]),
    )

    # ---- jaw / chin ----
    jaw_p = np.stack([x, (y + 0.55), (z + 0.05) * 1.05], axis=-1)
    jaw = sd_ellipsoid(jaw_p, np.array([0.65, 0.55, 0.80]))

    # ---- cheek fill (wraps skull + jaw so the transition is smooth) ----
    cheek = sd_ellipsoid(
        np.stack([x, y + 0.15, z + 0.05], axis=-1),
        np.array([0.78, 0.78, 0.88]),
    )

    head = smin(smin(skull, jaw, 0.25), cheek, 0.25)

    # ---- brow ridge (subtle bump across forehead) ----
    brow = sd_ellipsoid(
        np.stack([x, y - 0.35, z - 0.78], axis=-1),
        np.array([0.55, 0.12, 0.12]),
    )
    head = smin(head, brow, 0.15)

    # ---- nose (two stacked ellipsoids) ----
    nose_bridge = sd_ellipsoid(
        np.stack([x, y - 0.15, z - 0.82], axis=-1),
        np.array([0.08, 0.30, 0.18]),
    )
    nose_tip = sd_ellipsoid(
        np.stack([x, y + 0.08, z - 0.95], axis=-1),
        np.array([0.11, 0.10, 0.13]),
    )
    nose = smin(nose_bridge, nose_tip, 0.08)
    head = smin(head, nose, 0.06)

    # ---- eye sockets (subtract two spheres set slightly recessed) ----
    eyeL = sd_sphere(np.stack([x - 0.28, y - 0.18, z - 0.72], axis=-1), 0.16)
    eyeR = sd_sphere(np.stack([x + 0.28, y - 0.18, z - 0.72], axis=-1), 0.16)
    head = smax(head, -eyeL, 0.08)
    head = smax(head, -eyeR, 0.08)

    # ---- lip groove (shallow horizontal indent under nose) ----
    lip = sd_ellipsoid(
        np.stack([x, y + 0.28, z - 0.82], axis=-1),
        np.array([0.28, 0.04, 0.05]),
    )
    head = smax(head, -lip, 0.04)

    # ---- ears (small flattened ellipsoids on the sides) ----
    earL = sd_ellipsoid(
        np.stack([x - 0.92, y - 0.1, z + 0.05], axis=-1),
        np.array([0.07, 0.22, 0.17]),
    )
    earR = sd_ellipsoid(
        np.stack([x + 0.92, y - 0.1, z + 0.05], axis=-1),
        np.array([0.07, 0.22, 0.17]),
    )
    head = smin(head, earL, 0.06)
    head = smin(head, earR, 0.06)

    # ---- neck (capped cylinder below jaw) ----
    neck_y = y + 1.25
    neck_r = np.sqrt(x * x + (z + 0.05) ** 2) - 0.38
    # cap top/bottom
    neck = np.maximum(neck_r, np.abs(neck_y) - 0.45)
    head = smin(head, neck, 0.2)

    # ---- shoulders / clavicle stub so neck doesn't dangle ----
    shoulder = sd_capsule(
        p,
        np.array([-0.85, -1.65, 0.0]),
        np.array([0.85, -1.65, 0.0]),
        0.32,
    )
    head = smin(head, shoulder, 0.2)

    return head


def sdf_and_grad(p, eps=1e-3):
    """Finite-difference gradient of the SDF, vectorised."""
    e = np.array([[eps, 0, 0], [0, eps, 0], [0, 0, eps]])
    d0 = head_sdf(p)
    gx = head_sdf(p + e[0]) - head_sdf(p - e[0])
    gy = head_sdf(p + e[1]) - head_sdf(p - e[1])
    gz = head_sdf(p + e[2]) - head_sdf(p - e[2])
    grad = np.stack([gx, gy, gz], axis=-1) / (2 * eps)
    return d0, grad


def project_to_surface(pts, steps=6):
    """Walk each point toward the zero isosurface along -gradient."""
    for _ in range(steps):
        d, g = sdf_and_grad(pts)
        gn = np.linalg.norm(g, axis=-1, keepdims=True)
        gn = np.maximum(gn, 1e-6)
        pts = pts - g * (d[..., None] / gn)
    return pts


def main():
    rng = np.random.default_rng(7)

    # Fire candidates into a bbox that comfortably wraps the head+neck+shoulders.
    oversample = 6
    n_candidates = TARGET_POINTS * oversample
    candidates = rng.uniform(
        low=[-1.3, -2.1, -1.3],
        high=[1.3, 1.5, 1.3],
        size=(n_candidates, 3),
    )

    # Keep only candidates within a band around the surface (speeds up projection).
    d = head_sdf(candidates)
    keep = np.abs(d) < 0.25
    candidates = candidates[keep]
    print(f"candidates near surface: {len(candidates)}")

    # Project onto the actual zero-level set.
    pts = project_to_surface(candidates, steps=8)

    # Cull anything that didn't converge (still far from surface).
    final_d = head_sdf(pts)
    pts = pts[np.abs(final_d) < 0.015]
    print(f"converged on surface: {len(pts)}")

    # Thin to target count with a simple stride if we have too many.
    if len(pts) > TARGET_POINTS:
        idx = rng.choice(len(pts), TARGET_POINTS, replace=False)
        pts = pts[idx]

    # Normalise + scale to ~20 unit height (matches face3d_points.html camera).
    cmin = pts.min(axis=0)
    cmax = pts.max(axis=0)
    center = (cmin + cmax) / 2
    size = (cmax - cmin).max()
    scale = 20.0 / size
    pts = (pts - center) * scale

    flat = pts.reshape(-1).round(4).tolist()
    DST.write_text(json.dumps({"positions": flat, "count": len(pts)}))
    print(f"wrote {DST}  ({DST.stat().st_size // 1024} KB, {len(pts)} pts)")


if __name__ == "__main__":
    main()

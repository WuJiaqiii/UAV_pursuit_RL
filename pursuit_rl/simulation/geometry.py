"""NED world, FRD body. Rotation maps body vectors to world vectors."""

import numpy as np


def cross_matrix(w):
    w = np.asarray(w, dtype=np.float64)
    a = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    x, y, z = np.moveaxis(w, -1, 0)
    a[..., 0, 1] = -z
    a[..., 0, 2] = y
    a[..., 1, 0] = z
    a[..., 1, 2] = -x
    a[..., 2, 0] = -y
    a[..., 2, 1] = x
    return a


def exp_so3(theta):
    """Stable Rodrigues exponential, batch (...,3), radians."""
    theta = np.asarray(theta, dtype=np.float64)
    angle = np.linalg.norm(theta, axis=-1)
    a = np.empty_like(angle)
    b = np.empty_like(angle)
    small = angle < 1.0e-4
    z = angle[~small]
    a[~small] = np.sin(z) / z
    b[~small] = (1 - np.cos(z)) / (z * z)
    z = angle[small] ** 2
    a[small] = 1 - z / 6 + z * z / 120
    b[small] = 0.5 - z / 24 + z * z / 720
    k = cross_matrix(theta)
    return np.eye(3) + a[..., None, None] * k + b[..., None, None] * (k @ k)


def body_frame(rot, world_vec):
    return np.einsum("nji,nj->ni", rot, world_vec)


def world_frame(rot, body_vec):
    return np.einsum("nij,nj->ni", rot, body_vec)


def cpa(rel_p, rel_v, horizon=2.0):
    """CV closest approach on a bounded nonnegative time horizon."""
    speed2 = np.sum(rel_v * rel_v, axis=-1)
    time = np.clip(
        -np.sum(rel_p * rel_v, axis=-1) / np.maximum(speed2, 1e-9), 0, horizon
    )
    time = np.where(speed2 > 1e-8, time, 0.0)
    minimum = np.linalg.norm(rel_p + rel_v * time[:, None], axis=-1)
    return minimum, time


def segment_first_hit(r0, r1, radius):
    """First intersection of piecewise-linear relative trajectory with sphere.

    Returns hit and fraction in [0,1]. It does not establish the exact
    continuous-time intersection of nonlinear dynamics.
    """
    d = r1 - r0
    a = np.sum(d * d, axis=1)
    b = 2 * np.sum(r0 * d, axis=1)
    c = (
        np.sum(r0 * r0, axis=1) - radius * radius
        if np.isscalar(radius)
        else np.sum(r0 * r0, axis=1) - np.asarray(radius) ** 2
    )
    inside = c <= 0
    disc = b * b - 4 * a * c
    root = np.sqrt(np.maximum(disc, 0.0))
    t = (-b - root) / (2 * np.maximum(a, 1e-12))
    valid = (disc >= 0) & (a > 1e-12) & (t >= 0) & (t <= 1)
    return inside | valid, np.where(inside, 0.0, np.where(valid, t, 1.0))

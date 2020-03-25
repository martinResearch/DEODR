"""Tooling functions for 3D geometry processing with their backward derivatives implementations."""

import numpy as np


def qrot(q, v):
    if q.ndim == 2:
        uv = np.cross(q[:, None, :3], v[None, :, :])
        uuv = np.cross(q[:, None, :3], uv)
        vr = v + 2 * (q[:, None, [3]] * uv + uuv)
    else:
        uv = np.cross(q[:3], v)
        uuv = np.cross(q[:3], uv)
        vr = v + 2 * (q[3] * uv + uuv)

    return vr


def qrot_backward(q, v, vr_b):
    uv = np.cross(q[:3], v)
    v_b = vr_b.copy()
    q_b = np.zeros((4))
    q_b[3] = 2 * np.sum(vr_b * uv)
    uuv_b = 2 * vr_b.copy()
    uv_b = 2 * vr_b * q[3] + np.cross(uuv_b, q[:3])
    q_b[:3] = np.sum(np.cross(uv, uuv_b), axis=0) + np.sum(np.cross(v, uv_b), axis=0)
    v_b += np.cross(uv_b, q[:3])
    return q_b, v_b


def normalize(x, axis=-1):
    n2 = np.sum(x ** 2, axis=axis)
    n = np.sqrt(n2)
    xn = x / np.expand_dims(n, axis)
    return xn


def normalize_backward(x, xn_b, axis=-1):
    n2 = np.sum(x ** 2, axis=axis)
    n = np.sqrt(n2)
    inv_n = 1 / n
    n_b = -np.sum(xn_b * x, axis=axis) * (inv_n ** 2)
    x_b = (xn_b + x * np.expand_dims(n_b, axis)) * np.expand_dims(inv_n, axis)
    return x_b


def cross_backward(u, v, c_b):
    v_b = np.cross(c_b, u)
    u_b = np.cross(v, c_b)
    return u_b, v_b

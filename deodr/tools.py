"""Tooling functions for 3D geometry processing with their backward derivatives implementations."""

from typing import Callable, Tuple
import numpy as np


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    assert q.ndim in [1, 2]
    assert q.shape[-1] == 4
    assert v.ndim == 2
    assert v.shape[-1] == 3

    if q.ndim == 2:
        uv = np.cross(q[:, None, :3], v[None, :, :])
        uuv = np.cross(q[:, None, :3], uv)
        return v + 2 * (q[:, None, [3]] * uv + uuv)
    else:
        uv = np.cross(q[:3], v)
        uuv = np.cross(q[:3], uv)
        return v + 2 * (q[3] * uv + uuv)


def qrot_backward(
    q: np.ndarray, v: np.ndarray, vr_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    uv = np.cross(q[:3], v)
    v_b = vr_b.copy()
    q_b = np.zeros((4))
    q_b[3] = 2 * np.sum(vr_b * uv)
    uuv_b = 2 * vr_b.copy()
    uv_b = 2 * vr_b * q[3] + np.cross(uuv_b, q[:3])
    q_b[:3] = np.sum(np.cross(uv, uuv_b), axis=0) + np.sum(np.cross(v, uv_b), axis=0)
    v_b += np.cross(uv_b, q[:3])
    return q_b, v_b


def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n2 = np.sum(x**2, axis=axis)
    n = np.sqrt(n2)
    return x / np.expand_dims(n, axis)


def normalize_backward(x: np.ndarray, xn_b: np.ndarray, axis: int = -1) -> np.ndarray:
    n2 = np.sum(x**2, axis=axis)
    n = np.sqrt(n2)
    inv_n = 1 / n
    n_b = -np.sum(xn_b * x, axis=axis) * (inv_n**2)
    return (xn_b + x * np.expand_dims(n_b, axis)) * np.expand_dims(inv_n, axis)


def cross_backward(
    u: np.ndarray, v: np.ndarray, c_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    v_b = np.cross(c_b, u)
    u_b = np.cross(v, c_b)
    return u_b, v_b


def jacobian_finite_differences(
    func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:

    v0 = func(x)
    nx = x.copy()
    jacobian = np.zeros((v0.size, x.size))
    nx_flat = nx.ravel()
    for d in range(x.size):
        nx_flat[d] = nx_flat[d] + epsilon
        d1 = func(nx)
        nx_flat[d] = nx_flat[d] - epsilon
        d2 = func(nx)
        nx_flat[d] = x.flat[d]
        jacobian[:, d] = (d1 - d2).flatten() / (2 * epsilon)
    v02 = func(x)
    assert v0 == v02
    return jacobian


def check_jacobian_finite_differences(
    jac: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    epsilon: float = 1e-7,
    tol: float = 1e-4,
) -> None:
    nx = x.copy()
    nx_flat = nx.ravel()
    for d in range(x.size):
        nx_flat[d] = x.flat[d] + epsilon
        d1 = func(nx)
        nx_flat[d] = x.flat[d] - epsilon
        d2 = func(nx)
        nx_flat[d] = x.flat[d]
        jac_cold_fd = (d1 - d2).flatten() / (2 * epsilon)
        max_diff = np.max(np.abs(jac[..., d] - jac_cold_fd))
        assert max_diff < tol

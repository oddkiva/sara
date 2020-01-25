import numpy as np
import scipy.linalg as la


def rotation(axis, theta):
    return la.expm(np.cross(np.eye(3)), axis / la.norm(axis) * theta)


def skew(v):
    if len(v) == 4:
        v = v[:3] / v[3]
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T


def cofactors(E):
    return np.array([np.cross(E[:, (i + 1) % 3], E[:, (i + 2) % 3])
                     for i in range(3)])


def rodrigues(theta, v):
    V = skew(v)
    return np.eye(3) \
        + np.sin(theta) * V.dot(v) \
        + (1 - np.cos(theta)) * V.dot(V)

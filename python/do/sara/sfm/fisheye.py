import numpy as np

import pylab


def rectilinear(points, f):
    # Calculate the radial distance w.r.t. the optical center.
    tan_theta = points[:, 1] / points[:, 0]
    r = f * tan_theta
    r = r[:, np.newaxis]

    # Recover the unit orientation vector on which to apply the radial
    # transformation.
    theta = np.arctan2(points[:, 1], points[:, 0])
    u = np.stack((np.cos(theta), np.sin(theta))).T

    # The distorted points.
    return r * u

def equidistant(points, f):
    theta = np.arctan2(points[:, 1], points[:, 0])

    # Calculate the radial distance w.r.t. the optical center.
    r = f * theta
    r = r[:, np.newaxis]

    # Recover the unit orientation vector on which to apply the radial
    # transformation.
    u = np.stack((np.cos(theta), np.sin(theta))).T

    # The distorted points.
    return r * u

def opensfm_fisheye(points, f, k1=1e-1, k2=1e-1):
    # All the 2D points correspond to 3D points lying on the plane z = 1.
    x = points[:, 0]
    y = points[:, 1]
    r2 = x ** 2 + y ** 2
    r = np.sqrt(r2)
    theta = np.arctan(r)
    d = 1 + k1 * r2 + k2 * theta ** 4
    u = f * d * theta * x/r
    v = f * d * theta * y/r
    return np.stack((u, v)).T

def opensfm_spherical(points):
    # All the 2D points correspond to 3D points lying on the plane z = 1.
    x = points[:, 0]
    y = points[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    lon = np.arctan(x)
    lat = -np.arctan(-y/r)
    u = lon / (2 * np.pi)
    v = lat / (2 * np.pi)
    return np.stack((u, v)).T


# The points.
a1 = np.arange(-1, 1, 0.1, dtype=np.float)
a1 = np.stack((np.ones(len(a1)), a1)).T

a2 = np.arange(-1, 1, 0.1, dtype=np.float)
a2 = np.stack((a2, np.ones(len(a2)))).T


# The focal length.
f = 1.

# Fisheye rectilinear distortion model.
# a1_d = opensfm_fisheye(a1, f)
# a2_d = opensfm_fisheye(a2, f)
# pylab.scatter(a1[:, 0], a1[:, 1])
# pylab.scatter(a1_d[:, 0], a1_d[:, 1])
# pylab.scatter(a2[:, 0], a2[:, 1])
# pylab.scatter(a2_d[:, 0], a2_d[:, 1])

a = np.concatenate((a1, a2))
# a_d = opensfm_fisheye(a, f)
a_d = opensfm_spherical(a)
pylab.scatter(a[:, 0], a[:, 1])
pylab.scatter(a_d[:, 0], a_d[:, 1])



pylab.show()

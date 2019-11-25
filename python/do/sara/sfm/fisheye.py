import numpy as np

a = np.arange(-1, 1, 0.1, dtype=np.float)
a = np.stack((np.ones(len(a)), a)).T

f = 1.

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

a_distorted = rectilinear(a, f)

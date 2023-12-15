import numpy as np
from scipy.spatial.transform import Rotation

from do.sara.sfm.essential_matrix import camera_matrix


def rq(A):
    Q, R = np.linalg.qr(np.flipud(A).T)

    R = np.flipud(R.T)
    R = np.fliplr(R)

    Q = np.flipud(Q.T)

    return R, Q

def resectioning_hartley_zisserman(X, x):
    if X.shape[1] != x.shape[1]:
        raise ValueError('Invalid data!')

    n = X.shape[1]

    A = np.zeros((2 * n, 12))
    for i in range(n):
        Xi_T = X[:, i]
        ui, vi = x[:2, i]

        A[2 * i + 0,  :4] = Xi_T
        A[2 * i + 0, 8: ] = -ui * Xi_T

        A[2 * i + 1, 4:8] = Xi_T
        A[2 * i + 1, 8: ] = -vi * Xi_T

    _, _, Vh = np.linalg.svd(A)
    P_flat = Vh[-1, :]
    P = P_flat.reshape((3, 4))

    M = P[:, :3]
    K, R = rq(M)

    # Also the RQ factorization is not unique. Fortunately we want the
    # canonical form of the calibration matrix K where ax and ay are positive,
    # so let's flip the axes.

    # As explained by http://ksimek.github.io/2012/08/14/decompose/, we can use
    # Jan-Erik Solem's approach to flip the axes
    S = np.diag(np.sign(np.diag(K)))
    K = K @ S
    # Reorient the axes of the rotation matrix.
    R = S @ R

    # Recover the translation.
    t = np.linalg.inv(K) @ P[:, -1]
    t = t[:, np.newaxis]

    # Recall that the Frobenius norm of P is one, so we have reconstructed the
    # projection matrix only up to a scale.

    # The canonical form of the K matrix is:
    # [ax,  s, u0]
    # [ 0, ay, v0]
    # [ 0,  0,  1]

    # Thus the RQ factorization will recover K up to a scale factor.
    # So first thing first, rescale K and P as follows:
    scale = K[2, 2]
    K /= scale


    return K, R, t


def rot_z(angle):
    return Rotation.from_euler('z', angle, degrees=False).as_matrix()

def rot_y(angle):
    return Rotation.from_euler('y', angle, degrees=False).as_matrix()

def rot_x(angle):
    return Rotation.from_euler('x', angle, degrees=False).as_matrix()

def rigid_body_transform(R, t):
    T = camera_matrix(R, t)
    one = np.array([[0, 0, 0, 1]])
    return np.concatenate((T, one))

def project_to_film(P, X):
    x = P @ X
    x[:2, :] /= x[2, :]
    return x[:2, :]


def make_cube_vertices():
    X = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ], dtype=np.float)
    center = 0.5 * np.ones((3, 1), dtype=np.float)
    X = X - center

    ones = np.ones((1, X.shape[1]))

    X = np.concatenate((X, ones))

    return X

def make_relative_motion():
    R = rot_z(0.1) @ rot_x(0.2) @ rot_y(0.3)
    t = np.array([-2, -0.2, 10])[:, np.newaxis]
    return R, t

def make_camera_projection_matrix(R, t):
    return camera_matrix(R, t)


# Generate 3D points.
X = make_cube_vertices()
# Translate the cube vertices 10m further away from the camera center.
X[2,:] += 10

# Create a camera
R, t = make_relative_motion()
T = rigid_body_transform(R, t)
P = make_camera_projection_matrix(R, t)


np.set_printoptions(precision=2)
print('* World coordinates')
print('  X =\n', X)

Xc = T @ X
print('* Camera coordinates')
print('  Xc =\n', Xc)

x = project_to_film(P, X)
print('* Film coordinates')
print('  x =\n', x)

print('* Camera resectioning...')
K1, R1, t1 = resectioning_hartley_zisserman(X, x)
print('  K1 =\n', K1)
print('  R1 =\n', R1)
print('  t1 =\n', t1)


P1 = K1 @ make_camera_projection_matrix(R1, t1)
print('  P1 =\n', P1)
print('  |P1 - P| =', np.linalg.norm(P - P1))


import IPython; IPython.embed()

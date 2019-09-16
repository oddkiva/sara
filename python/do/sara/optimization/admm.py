import numpy as np
import scipy.linalg as la


def admm(A, b, num_iter=1000, eps=1e-8, rho=1):
    """
    Solves min_x |Ax - b|_1

    This is equivalent to minimizing the objective function:
        f = lambda x: 0
        g = lambda z: np.abs(z)  # L1-norm of z.
        objective = lambda z: np.abs(z)
    subject to:
        constraint = lambda (x, z): A.dot(x) - z - b

    Its augmented Lagrangian is:
        L = lambda (x, z, y, rho): objective(x, z) \
            + y.T.dot(constraint(x, z))
            + rho / 2. * constraint(x, z).T.dot(constraint(x, z))

    In practice we use the scaled dual form, by changing variable.
        u = 1. / rho * y

    The dual scaled form of the constraint is:
        constraint = lambda (x, z, u): A.dot(x) - z - b + u

    Modulo a constant the augmented Lagrangian is equivalent to:
        L = lambda (x, z, u, rho): objective(z) \
            + rho / 2. * constraint(x, z, u).T.dot(constraint(x, z, u))
    """

    # Soft thresholding operator.
    soft_thres = lambda (x, a): la.sign(x) * la.max(la.abs(x) - a, 0)

    # Functions to update (x, z, u).
    x_next = lambda (z0, u0): la.lstsq(A.T.dot(A), A.T.dot(z0 + b - u0))[0]
    z_next = lambda (x1, u0): soft_thres(A.dot(x1) - b + u0, 1. / rho)
    u_next = lambda (u0, x1, z1): u0 + A.dot(x1) - z1 - b

    # Initialize (x0, z0, u0).
    x0 = x
    z0 = A.dot(x0) - b
    u0 = np.zeros(b.shape)

    x1 = None
    z1 = None
    u1 = None

    delta_x = np.inf
    n = 0
    while delta_x > eps and n < num_iter:
        # Update the variables.
        x1 = x_next(z0, u0)
        z1 = z_next(x1, u0)
        u1 = u_next(u0, x1, z1)

        # Evaluate the distance
        delta_x = la.norm(x1 - x0)
        n += 1

        # Update the variable for the next iteration.
        x0 = x1
        z0 = z1
        u0 = u1

    return x1

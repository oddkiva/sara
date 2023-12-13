import numpy as np

import scipy.linalg as la

import sympy as sp

from do.sara.sfm.essential_matrix import extract_epipoles


def triangulate_longuet_higgins(R, t, left, right):
    """
    Works only if correspoinding point lie exactly in the epipolar lines.
    So apply:
    - Hartley-Sturm triangulation method
    - Lindstrom triangulation method
    """
    num_points = left.shape[1]
    z = (R[0, :].dot(t) * np.ones(num_points) - R[2, :].dot(t) * right[0, :]) \
        / (R[0, :].dot(left) - R[2, :].dot(left) * right[0, :])

    x = left[0, :] * z
    y = left[1, :] * z

    return np.array([x, y, z, np.ones(num_points)])


def x_axis_rigid_alignment(x, e):
    """
    Rotate and translate the coordinate system of the film plane
    so that:
    - the point x is lying at the origin of the coordinate system
    - the epipole e is lying on the positive side of the x-axis.
    """
    delta = e - x
    theta = -np.arctan2(delta[1], delta[0])
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [            0,              0, 1]])
    T = np.eye(3)
    T[:2, 2] = -x[:2]
    return R.dot(T)


def expand_hartley_sturm_poly():
    a, b, c, d, fl, fr, t = sp.symbols('a b c d fl fr t')
    r = t * ((a * t + b) ** 2 + fr ** 2 * (c * t + d) ** 2) ** 2 \
        - (a * d - b * c) * (1 + t ** 2 * fl ** 2) ** 2 * (a * t + b) * (c * t + d)
    r_expanded = sp.expand(r)
    r_poly = sp.Poly(r_expanded, t)

    for i in range(r_poly.degree() + 1):
        print(i)
        print(r_poly.coeff_monomial(t ** i))

    return r_poly


def expand_hartley_sturm_poly_abs():
    a, b, c, d, fl, fr, t = sp.symbols('a b c d fl fr t')
    A = 1
    B = (a * d - b * c) ** 2 * (a * t + b) ** 2
    C = (1 + t**2 * fl**2) ** 3
    D = ((a*t + b) ** 2 + fr**2 * (c*t + d)**2) ** 3
    r = A * D - B * C

    r_expanded = sp.expand(r)
    r_poly = sp.Poly(r_expanded, t)

    for i in range(r_poly.degree() + 1):
        print(i)
        print(r_poly.coeff_monomial(t ** i))

    return r_poly


def poly_hartley_sturm(a, b, c, d, fl, fr):
    r_coeff = np.array([
        -a*b*d**2 + b**2*c*d,
        # t
        -a**2*d**2 + b**4 + b**2*c**2 + 2*b**2*d**2*fr**2 + d**4*fr**4,
        # t ** 2
        -a**2*c*d + 4*a*b**3 + a*b*c**2 - 2*a*b*d**2*fl**2 + 4*a*b*d**2*fr**2
        + 2*b**2*c*d*fl**2 + 4*b**2*c*d*fr**2 + 4*c*d**3*fr**4,
        # t ** 3
        6*a**2*b**2 - 2*a**2*d**2*fl**2 + 2*a**2*d**2*fr**2 + 8*a*b*c*d*fr**2
        + 2*b**2*c**2*fl**2 + 2*b**2*c**2*fr**2 + 6*c**2*d**2*fr**4,
        # t ** 4
        4*a**3*b - 2*a**2*c*d*fl**2 + 4*a**2*c*d*fr**2 + 2*a*b*c**2*fl**2 +
        4*a*b*c**2*fr**2 - a*b*d**2*fl**4 + b**2*c*d*fl**4 + 4*c**3*d*fr**4,
        # t ** 5
        a**4 + 2*a**2*c**2*fr**2 - a**2*d**2*fl**4 + b**2*c**2*fl**4 +
        c**4*fr**4,
        # t ** 6
        -a**2*c*d*fl**4 + a*b*c**2*fl**4
    ])
    r_coeff /= r_coeff[-1]
    return r_coeff


def poly_abs_hartley_sturm(a, b, c, d, fl, fr):
    r_coeff = np.array([
        # 0
        -a**2*b**2*d**2 + 2*a*b**3*c*d + b**6 - b**4*c**2 + 3*b**4*d**2*fr**2 +
        3*b**2*d**4*fr**4 + d**6*fr**6,
        # 1
        -2*a**3*b*d**2 + 4*a**2*b**2*c*d + 6*a*b**5 - 2*a*b**3*c**2 +
        12*a*b**3*d**2*fr**2 + 6*a*b*d**4*fr**4 + 6*b**4*c*d*fr**2 +
        12*b**2*c*d**3*fr**4 + 6*c*d**5*fr**6,
        # 2
        -a**4*d**2 + 2*a**3*b*c*d + 15*a**2*b**4 - a**2*b**2*c**2 -
        3*a**2*b**2*d**2*fl**2 + 18*a**2*b**2*d**2*fr**2 + 3*a**2*d**4*fr**4 +
        6*a*b**3*c*d*fl**2 + 24*a*b**3*c*d*fr**2 + 24*a*b*c*d**3*fr**4 -
        3*b**4*c**2*fl**2 + 3*b**4*c**2*fr**2 + 18*b**2*c**2*d**2*fr**4 +
        15*c**2*d**4*fr**6,
        # 3
        20*a**3*b**3 - 6*a**3*b*d**2*fl**2 + 12*a**3*b*d**2*fr**2 +
        12*a**2*b**2*c*d*fl**2 + 36*a**2*b**2*c*d*fr**2 + 12*a**2*c*d**3*fr**4
        - 6*a*b**3*c**2*fl**2 + 12*a*b**3*c**2*fr**2 + 36*a*b*c**2*d**2*fr**4 +
        12*b**2*c**3*d*fr**4 + 20*c**3*d**3*fr**6,
        # 4
        15*a**4*b**2 - 3*a**4*d**2*fl**2 + 3*a**4*d**2*fr**2 +
        6*a**3*b*c*d*fl**2 + 24*a**3*b*c*d*fr**2 - 3*a**2*b**2*c**2*fl**2 +
        18*a**2*b**2*c**2*fr**2 - 3*a**2*b**2*d**2*fl**4 +
        18*a**2*c**2*d**2*fr**4 + 6*a*b**3*c*d*fl**4 + 24*a*b*c**3*d*fr**4 -
        3*b**4*c**2*fl**4 + 3*b**2*c**4*fr**4 + 15*c**4*d**2*fr**6,
        # 5
        6*a**5*b + 6*a**4*c*d*fr**2 + 12*a**3*b*c**2*fr**2 -
        6*a**3*b*d**2*fl**4 + 12*a**2*b**2*c*d*fl**4 + 12*a**2*c**3*d*fr**4 -
        6*a*b**3*c**2*fl**4 + 6*a*b*c**4*fr**4 + 6*c**5*d*fr**6,
        # 6
        a**6 + 3*a**4*c**2*fr**2 - 3*a**4*d**2*fl**4 + 6*a**3*b*c*d*fl**4 -
        3*a**2*b**2*c**2*fl**4 - a**2*b**2*d**2*fl**6 + 3*a**2*c**4*fr**4 +
        2*a*b**3*c*d*fl**6 - b**4*c**2*fl**6 + c**6*fr**6,
        # 7
        -2*a**3*b*d**2*fl**6 + 4*a**2*b**2*c*d*fl**6 - 2*a*b**3*c**2*fl**6,
        # 8
        -a**4*d**2*fl**6 + 2*a**3*b*c*d*fl**6 - a**2*b**2*c**2*fl**6
    ])
    r_coeff /= r_coeff[-1]
    return r_coeff


def lambda_l(t, fl):
    return np.array([t * fl, 1, -t])


def lambda_r(t, a, b, c, d, fr):
    return np.array([-fr * (c * t + d), a * t + b, c * t + d])


def err_l(t, fl):
    return t ** 2 / (1 + t ** 2 * fl ** 2)


def err_r(t, a, b, c, d, fr):
    return (c*t + d) ** 2 / ((a * t + b) ** 2 + fr ** 2 * (c * t + d) ** 2)


def reproj_err(t, a, b, c, d, fl, fr):
    return err_l(t, fl) + err_r(t, a, b, c, d, fr)


def triangulate_hartley_sturm(E, el, er, xl, xr, method='poly_abs'):
    """ Retrieve the 3D points from a set of point correspondences.

    Hartley-Sturm's method finds the **globally** optimal 3D point but it is
    not direct because we require finding the roots of a polynomial and finding
    roots accurately is going to be iterative anyways.

    Standard methods to compute the roots of a polynomial is done via
    calculating the SVD of companion matrix or applying Jenkins-Traub RPOLY
    algorithm.
    """
    Tl = x_axis_rigid_alignment(xl, el / el[-1])
    Tr = x_axis_rigid_alignment(xr, er / er[-1])

    fl = 1. / Tl.dot(el / el[-1])[0]
    fr = 1. / Tr.dot(er / er[-1])[0]

    Tl_inv = la.inv(Tl)
    Tr_inv = la.inv(Tr)

    E1 = Tr_inv.T.dot(E).dot(Tl_inv)
    origin = np.array([0, 0, 1])
    el1 = np.array([1, 0, fl])
    er1 = np.array([1, 0, fr])
    print('0.T E1 0 = ', np.abs(origin.dot(E1).dot(origin)))
    print('er1.T E1 el1 = ', np.abs(er1.dot(E1).dot(el1)))

    a = E1[1, 1]
    b = E1[1, 2]
    c = E1[2, 1]
    d = E1[2, 2]
    F = np.array([[fl * fr * d, -fr * c, -fr * d],
                  [-fl * b, a, b],
                  [-fl * d, c, d]])
    assert la.norm(F - E1) / la.norm(E1) < 1e-12

    if method == 'poly':
        r_coeff = poly_hartley_sturm(a, b, c, d, fl, fr)
    else:
        r_coeff = poly_abs_hartley_sturm(a, b, c, d, fl, fr)

    roots = np.roots(r_coeff)
    ts = [r.real for r in roots if r.imag == 0]
    print('roots =', roots)
    print('ts = ', ts)

    # Minimize the univariate polynomial d(u, lambda(t)) + d(u', lambda'(t)) in t.
    # Find the zero of the derivative.
    errs = np.array([reproj_err(t, a, b, c, d, fl, fr) for t in ts])
    i = np.argmin(errs)
    ti = ts[i]
    print('t = ', ti)

    line_l = lambda_l(ti, fl)
    line_r = lambda_r(ti, a, b, c, d, fr)
    assert np.abs(line_l.dot(el1)) == 0
    assert np.abs(line_r.dot(er1)) == 0

    nl = np.hstack((line_l[:2], 0))
    nl /= la.norm(nl)
    nr = np.hstack((line_r[:2], 0))
    nr /= la.norm(nr)

    normal_l = (origin - el1/ el1[-1]).dot(nl) * nl
    normal_r = (origin - er1/ er1[-1]).dot(nr) * nr
    print('dx_r.T E1 dxl = ', (origin - normal_r).dot(E1).dot(origin - normal_l))

    xl1 = Tl_inv.dot(origin - normal_l)
    xr1 = Tr_inv.dot(origin - normal_r)
    print('xr1.dot(E).dot(xl1) = ', np.abs(xr1.dot(E).dot(xl1)))
    assert np.abs(xr1.dot(E).dot(xl1)) < 1e-6

    return xl1, xr1


def triangulate_lindstrom_iterative(E, xl, xr, K=2):
    """UNTESTED."""
    S = np.array([[1, 0, 0],
                  [0, 1, 0]])
    xl0 = xl.copy()
    xr0 = xr.copy()
    xl1 = xl.copy()
    xr1 = xr.copy()

    nl0 = S.dot(E).dot(xl0)
    nr0 = S.dot(E.T).dot(xr0)

    c1 = xr0.dot(E).dot(xl0)

    for k in range(K):
        nl1 = S.dot(E).dot(xl0)
        nr1 = S.dot(E.T).dot(xr0)

        a1 = nr1.dot(E).dot(nl1)
        b1 = 0.5 * (nl0.dot(nl1) + nr0.dot(nr1))
        d1 = np.sqrt(b1 * b1 - a1 * c1)

        l1 = c1 / (b1 + np.sign(b1) * d1)

        dxr1 = l1 * nr1
        dxl1 = l1 * nl1

        xl1 = xl0 - S.T.dot(dxl1)
        xr1 = xr0 - S.T.dot(dxr1)

        xl0 = xl1
        xr0 = xr1

    return (xl1, xr1)


def triangulate_lindstrom_two_iterations(E, xl, xr):
    """UNTESTED."""
    S = np.array([[1, 0, 0],
                  [0, 1, 0]])
    E1 = S.dot(E).dot(S.T)

    nl = S.dot(E).dot(xl)
    nr = S.dot(E.T).dot(xr)

    a = nr1.dot(E).dot(nl)
    b = 0.5 * (nl.dot(nl) + nr.dot(nr))
    c = xr.dot(E).dot(xl)
    d = np.sqrt(b * b - a * c)
    l = c / (b + d)

    dxl = l * nl
    dxr = l * nr

    nr -= E1.dot(dxl)
    nl -= E1.T.dot(dxr)

    l = l * (2 * d) / (nr.dot(nr) + nl.dot(nl))
    dxl = l * nl
    dxr = l * nr

    xl1 = xl - S.T.dot(dxl)
    xr1 = xr - S.T.dot(dxr)
    return (xl1, xr1)


def test():
    X = np.array([[-1.49998, -0.582700, -1.405910,  0.369386,  0.161931],
                  [-1.23692, -0.434466, -0.142271, -0.732996, -1.430860],
                  [ 1.51121,  0.437918,  1.358590,  1.038830,  0.106923],
                  [ 1      ,  1       ,  1       ,  1       ,  1       ]])

    R = np.array([[0.930432, -0.294044,   0.218711],
                  [0.308577,  0.950564, -0.0347626],
                  [ -0.197677, 0.0998334, 0.97517]])

    t = np.array([[0.1],
                  [0.2],
                  [0.3]])
    E = essential_matrix(R, t)
    C1 = camera_matrix(np.eye(3), np.zeros((3, 1)))
    C2 = camera_matrix(R, t)

    x1 = project(X, C1)
    x2 = project(X, C2)
    for i in range(5):
        print('[', i, ']  xr.dot(E).dot(xl) = ', x2[:, i].dot(E).dot(x1[:, i]))

    benchmark_relative_motion_extraction_method(E, R, t)
    X_est = triangulate_longuet_higgins(R, t, x1, x2)
    print("X =\n", X)
    print("X_est =\n", X_est)
    print('3D point relative estimation error = ', la.norm(X_est - X) / la.norm(X))
    return

    el, er = extract_epipoles(E)
    assert la.norm(er / er[-1] - t.flatten() / t[-1]) / la.norm(t / t[-1]) < 1e-12
    tr1 = R.T.dot(t).flatten()
    assert la.norm(el / el[-1] - tr1  / tr1[-1]) / la.norm(tr1 / tr1[-1]) < 1e-6
    assert np.abs(er.dot(E).dot(el)) < 1e-12
    for i in range(5):
        print('[', i, ']')
        xl0 = x1[:, i]
        xr0 = x2[:, i]

        print('BEFORE TRIANGULATION')
        print('xr0.dot(E).dot(xl0) = ', xr0.dot(E).dot(xl0))

        print('TRIANGULATING...')
        xl, xr = triangulate_hartley_sturm(E, el, er, xl0, xr0)

        print('AFTER TRIANGULATION')
        print('xr.dot(E).dot(xl) = ', xr.dot(E).dot(xl))
        print('xl0 =', xl0)
        print('xl1 =', xl)
        print('xr0 =', xr0)
        print('xr1 =', xr)
        print('')


if __name__ == '__main__':
    test()

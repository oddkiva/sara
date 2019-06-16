import numpy as np
import scipy.linalg as la
import sympy as sp


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


def essential_matrix(R, t):
    return skew(t).dot(R)
    #return R.dot(skew(t))


def camera_matrix(R, t):
    return np.hstack((R, t))

def project(X, P):
    x = P.dot(X)
    x /= x[2, :]
    return x


def extract_relative_motions_horn(E):
    """ http://people.csail.mit.edu/bkph/articles/Essential.pdf """
    EEt = E.dot(E.T)

    cofE = cofactors(E)
    norm_cofE = np.array([la.norm(cofE[i, :]) for i in range(3)])
    i = np.argmax(norm_cofE)

    ta = cofE[i, :] / norm_cofE[i] * np.sqrt(0.5 * EEt.trace())
    ta = np.reshape(ta, (3, 1))
    tb = -ta

    ta_sq_norm = ta.T.dot(ta)

    Ra = (cofE.T - skew(ta).dot(E)) / ta_sq_norm

    F = 2. * ta.dot(ta.T) / ta_sq_norm - np.eye(3)
    Rb = F.dot(Ra)

    return ((Ra, Rb), (ta, tb))


def extract_relative_motions_hartley_zisserman(E):
    U, S, Vt = la.svd(E)

    if la.det(U) < 0:
        U[:, 2] *= -1
    if la.det(Vt.T) < 0:
        Vt[2, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    Ra = U.dot(W).dot(Vt)
    Rb = U.dot(W.T).dot(Vt)

    ta = np.reshape(U[:, 2], (3, 1))
    tb = -ta

    return ((Ra, Rb), (ta, tb))


def extract_epipoles(F):
    U, _, Vt = la.svd(F)
    el = Vt[2, :]
    er = U[:, 2]
    return (el, er)


def benchmark_relative_motion_extraction_method(E):
    horn_method = extract_relative_motions_horn
    ((Ra, Rb), (ta, tb)) = horn_method(E)
    norm_ta = la.norm(ta)
    norm_tb = la.norm(tb)
    norm_t = la.norm(t)
    print('HORN')
    print("err_rel(ta, t) = ",
          la.norm(ta / norm_ta - t / norm_t) / la.norm(t / norm_t))
    print("err_rel(tb, t) = ",
          la.norm(tb / norm_ta - t / norm_t) / la.norm(t / norm_t))
    print("err_rel(Ra, R) = ", la.norm(Ra - R) / la.norm(R))
    print("err_rel(Rb, R) = ", la.norm(Rb - R) / la.norm(R))

    print('HARTLEY-ZISSERMAN')
    hz_method = extract_relative_motions_hartley_zisserman
    ((Ra, Rb), (ta, tb)) = hz_method(E)
    norm_ta = la.norm(ta)
    norm_tb = la.norm(tb)
    norm_t = la.norm(t)
    print("err_rel(ta, t) = ",
          la.norm(ta / norm_ta - t / norm_t) / la.norm(t / norm_t))
    print("err_rel(tb, t) = ",
          la.norm(tb / norm_ta - t / norm_t) / la.norm(t / norm_t))
    print("err_rel(Ra, R) = ", la.norm(Ra - R) / la.norm(R))
    print("err_rel(Rb, R) = ", la.norm(Rb - R) / la.norm(R))

    # Benchmarking with
    # % timeit for _ in range(100): horn_method(E)
    # % timeit for _ in range(100): hz_method(E)
    #
    # The two methods show the same order of accuracy.
    #
    # The SVD-based traditional method of extracting R and t is 2.35 times faster.
    # That's quite a surprise.
    #
    # However, Horn's method can retrieve the baseline (t) exactly! That can be
    # a real advantage.


def triangulate_longuet_higgins(R, t, left, right):
    num_points = left.shape[1]
    z = (R[0, :].dot(t) * np.ones(num_points) - R[2, :].dot(t) * right[0, :]) / \
        (R[0, :].dot(left) - R[2, :].dot(left) * right[0, :])

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
    D = ((a*t + b) ** 2 + fr**2 * (c*t + d)**2)** 3
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

def triangulate_hartley_sturm(el, er, xl, xr, method='poly_abs'):
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

benchmark_relative_motion_extraction_method(E)
X_est = triangulate_longuet_higgins(R, t, x1, x2)
print('3D point relative estimation error = ', la.norm(X_est - X) / la.norm(X))


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
    xl, xr = triangulate_hartley_sturm(el, er, xl0, xr0)

    print('AFTER TRIANGULATION')
    print('xr.dot(E).dot(xl) = ', xr.dot(E).dot(xl))
    print('xl0 =', xl0)
    print('xl1 =', xl)
    print('xr0 =', xr0)
    print('xr1 =', xr)
    print('')

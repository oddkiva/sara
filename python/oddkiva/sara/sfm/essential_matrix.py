import numpy as np
import scipy.linalg as la

from do.sara.sfm.geometry import cofactors, skew


def essential_matrix(R, t):
    return skew(t).dot(R)


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


def benchmark_relative_motion_extraction_method(E, R, t):
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
          la.norm(tb / norm_tb - t / norm_t) / la.norm(t / norm_t))
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

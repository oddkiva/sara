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

E = np.matmul(skew(t), R)

def horn_method(E):
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


def hartley_zisserman_method(E):
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
hz_method = hartley_zisserman_method
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

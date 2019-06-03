""" Li and Hartley's method.  """

import sympy as sp
from sympy.abc import x, y, z

X = sp.Matrix(sp.symbols(
    ' '.join( ['X{}'.format(i) for i in range(9)]))).reshape(3, 3)
Y = sp.Matrix(sp.symbols(
    ' '.join(['Y{}'.format(i) for i in range(9)]))).reshape(3, 3)
Z = sp.Matrix(sp.symbols(
    ' '.join(['Z{}'.format(i) for i in range(9)]))).reshape(3, 3)
W = sp.Matrix(
    sp.symbols(' '.join(['W{}'.format(i) for i in range(9)]))).reshape(3, 3)

# Decompose the essential matrix into a 4 bases of the Null space.
E = x * X + y * Y + z * Z + W

# Write the equations.
a = sp.det(E)
b = 2 * E * E.T * E.trace() - (E * E.T).trace() * E
CX = [a] + sp.flatten(b)
CX = sp.Matrix([sp.Poly(c, x, y, z) for c in CX])

# The vector X which depends only on variables (x, y).
X_ = sp.Matrix([x**3, y**3, x**2 * y, x * y**2, x**2 * z, x**2, y**2 * z, y**2,
                #
                x * y * z, x * y,
                #
                x, x * z, x * z**2,
                #
                y, y * z, y * z**2,
                #
                1, z, z**2, z**3])

# The vector A which depends only on the variable 'z'.
A = []
for i in range(10):
    Ai = []
    for j in range(20):
        Ai.append(CX[i].coeff_monomial(X_[j]))
    A.append(Ai)
A = sp.Matrix(A)

# Check that C * X_ == CX.
D = C * X_
diff = D - CX
diff = all([diff[i] in range(10)])

L, U, _ = A.LUdecomposition()

e = U[4, :]
f = U[5, :]
k = e - z * f





# # The vector X which depends only on variables (x, y).
# X_ = sp.Matrix([x**3, y**3, x**2 * y, x * y**2, x**2, y**2, x*y, x, y, 1])
#
# # The vector C which depends only on the variable 'z'.
# C = []
# for i in range(10):
#     Ci = []
#     for j in range(10):
#         Ci.append(CX[i].coeff_monomial(X_[j]))
#
#     Ci[4] += CX[i].coeff_monomial(x**2*z) * z
#     Ci[5] += CX[i].coeff_monomial(y**2*z) * z
#     Ci[6] += CX[i].coeff_monomial(x*y*z) * z
#     Ci[7] += CX[i].coeff_monomial(x*z) * z
#     Ci[8] += CX[i].coeff_monomial(y*z) * z
#     Ci[9] += CX[i].coeff_monomial(z) * z
#
#     Ci[7] += CX[i].coeff_monomial(x*z**2) * z**2
#     Ci[8] += CX[i].coeff_monomial(y*z**2) * z**2
#     Ci[9] += CX[i].coeff_monomial(z**2) * z**2
#
#     Ci[9] += CX[i].coeff_monomial(z**3) * z**3
#     C.append(Ci)
# C = sp.Matrix(C)
#
# # Check that C * X_ == CX.
# D = C * X_
# diff = D - CX
# diff = all([diff[i] in range(10)])
#
# # Calculate the determinant.
# det_C = sp.Poly(C.det_LU_decomposition(), z)

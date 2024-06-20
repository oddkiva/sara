import sympy as sp
from sympy.abc import x, y, z
from sympy.codegen.rewriting import create_expand_pow_optimization

from pathlib import Path


def get_sara_src_dir_path():
    idx = __file__.find("oddkiva/sara") + len("oddkiva/sara")
    sara_src_dir_path = __file__[:idx]
    return Path(sara_src_dir_path)


def build_essential_matrix_contraints(P, Q, M):
    rows = []

    row_0 = []
    for j in range(20):
        row_0.append(Q.coeff_monomial(M[j]))
    rows.append(row_0)

    for a in range(3):
        for b in range(3):
            P_ab = sp.simplify(P[a, b].as_poly(x, y, z))
            row_i = []
            for j in range(20):
                row_i.append(P_ab.coeff_monomial(M[j]))
            rows.append(row_i)

    return sp.Matrix(rows)


def form_determinant_of_resultant_matrix(S, M):
    def to_poly(P, i):
        M_reduced = M[10:]
        expr = P[i, 0] * M_reduced[0]
        for j in range(1, 10):
            expr += S[i, j] * M_reduced[j]
        return expr.as_poly()

    # Elimination tricks by Nister.
    e = to_poly(S, 0)
    f = to_poly(S, 1)
    g = to_poly(S, 2)
    h = to_poly(S, 3)
    i = to_poly(S, 4)
    j = to_poly(S, 5)

    k = (e - z * f).as_poly(x, y, z)
    l = (g - z * h).as_poly(x, y, z)
    m = (i - z * j).as_poly(x, y, z)

    # Key observation:
    # [x, y, 1].T lives in the nullspace Null(B),
    # where the coefficients of B are polynomials in z.
    #
    # Split the coeffs.
    B00 = sum([k.coeff_monomial(x * z ** i) * z ** i for i in range(0, 4)])
    B01 = sum([k.coeff_monomial(y * z ** i) * z ** i for i in range(0, 4)])
    B02 = sum([k.coeff_monomial(1 * z ** i) * z ** i for i in range(0, 5)])

    B10 = sum([l.coeff_monomial(x * z ** i) * z ** i for i in range(0, 4)])
    B11 = sum([l.coeff_monomial(y * z ** i) * z ** i for i in range(0, 4)])
    B12 = sum([l.coeff_monomial(1 * z ** i) * z ** i for i in range(0, 5)])

    B20 = sum([m.coeff_monomial(x * z ** i) * z ** i for i in range(0, 4)])
    B21 = sum([m.coeff_monomial(y * z ** i) * z ** i for i in range(0, 4)])
    B22 = sum([m.coeff_monomial(1 * z ** i) * z ** i for i in range(0, 5)])

    return sp.Matrix([[B00, B01, B02],
                      [B10, B11, B12],
                      [B20, B21, B22]])


def calculate_determinant(B):
    # Calculate the determinant minors.
    # They are also univariate polynomial w.r.t. variable z.
    p0 = B[0, 1] * B[1, 2] - B[0, 2] * B[1, 1]
    p1 = B[0, 2] * B[1, 0] - B[0, 0] * B[1, 2]
    p2 = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]

    # Finally the determinant polynomial in the variable z.
    n = (p0 * B[2, 0] + p1 * B[2, 1] + p2 * B[2, 2]).as_poly(z)

    return n, (p0, p1, p2)


# The 4 eigenvectors that span the nullspace.
X = sp.MatrixSymbol('X', 3, 3)
Y = sp.MatrixSymbol('Y', 3, 3)
Z = sp.MatrixSymbol('Z', 3, 3)
W = sp.MatrixSymbol('W', 3, 3)

# The essential matrix lives in the nullspace.
E = sp.Matrix(x * X + y * Y + z * Z + W)

# Auxiliary variable.
EEt = sp.Matrix(E * E.transpose())

# M = the following list of monomials enumerated in the order below.
M = [
    x ** 3,
    y ** 3,
    x ** 2 * y,
    x * y ** 2,
    x ** 2 * z,
    x ** 2,
    y ** 2 * z,
    y ** 2,
    x * y * z,
    x * y,
    #
    x,
    x * z,
    x * z ** 2,
    #
    y,
    y * z,
    y * z ** 2,
    #
    1,
    z,
    z ** 2,
    z ** 3
]

# Constraints
P = sp.Matrix(EEt * E) - EEt.trace() / 2 * E
Q = sp.simplify(E.det().as_poly(x, y, z))

A = build_essential_matrix_contraints(P, Q, M)

expand_opt = create_expand_pow_optimization(3)
e_constraints_file_path = (
    get_sara_src_dir_path() / "cpp" / "src" / "DO" / "Sara" /
        "MultiViewGeometry" / "MinimalSolvers" / "Precomputed" /
        "EssentialMatrixPolynomialConstraints.hpp"
)
with open(e_constraints_file_path, "w") as f:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_ij = expand_opt(A[i, j])
            code_ij = sp.cxxcode(A_ij, assign_to=f"A({i}, {j})")
            f.write(f"{code_ij}\n")

# Perform Gauss-Jordan elimination so that A(x, y, z) = 0
# simplifies as a system [[I, R], = [[0],
#                         [0, S]] =  [0]]
S = sp.MatrixSymbol('S', 6, 10)
B = form_determinant_of_resultant_matrix(S, M)
n, p = calculate_determinant(B)

resulting_determinant_file_path = (
    get_sara_src_dir_path() / "cpp" / "src" / "DO" / "Sara" /
        "MultiViewGeometry" / "MinimalSolvers" / "Precomputed" /
        "EssentialMatrixResultingDeterminant.hpp"
)
with open(resulting_determinant_file_path, "w") as f:
    n = expand_opt(n)
    for i in range(10):
        code_i = sp.cxxcode(n.coeff_monomial(z ** i), assign_to=f"n[{i}]")
        f.write(f"{code_i}\n")


import IPython; IPython.embed()

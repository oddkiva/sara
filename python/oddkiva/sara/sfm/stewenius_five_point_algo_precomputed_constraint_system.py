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
    x * x * x,
    x * x * y,
    x * y * y,
    y * y * y,
    x * x * z,
    x * y * z,
    y * y * z,
    x * z * z,
    y * z * z,
    z * z * z,
    x * x,
    x * y,
    y * y,
    x * z,
    y * z,
    z * z,
    x,
    y,
    z,
    1
]

# Constraints
P = sp.Matrix(EEt * E) - EEt.trace() / 2 * E
Q = sp.simplify(E.det().as_poly(x, y, z))

A = build_essential_matrix_contraints(P, Q, M)

mvg_src_dir_path = (get_sara_src_dir_path() / "cpp" / "src" / "DO" / "Sara" / "MultiViewGeometry")
stewenius_src_dir_path = mvg_src_dir_path / "MinimalSolvers" / "Stewenius"

expand_opt = create_expand_pow_optimization(3)
e_constraints_file_path = stewenius_src_dir_path / "EssentialMatrixPolynomialConstraints.hpp"
with open(e_constraints_file_path, "w") as f:
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_ij = expand_opt(A[i, j])
            code_ij = sp.cxxcode(A_ij, assign_to=f"A({i}, {j})")
            f.write(f"{code_ij}\n")

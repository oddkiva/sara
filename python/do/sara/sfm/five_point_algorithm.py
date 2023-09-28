import sympy as sp
from sympy.abc import x, y, z

# Solve the cubic polynomial.
X = sp.MatrixSymbol('X', 3, 3)
Y = sp.MatrixSymbol('Y', 3, 3)
Z = sp.MatrixSymbol('Z', 3, 3)
W = sp.MatrixSymbol('W', 3, 3)

# Form the symbolic matrix expression as reported in the paper.
E = sp.Matrix(x * X + y * Y + z * Z + W)

essential_constraint = E * E.T * E - sp.Trace(E * E.T) * E
rank_2_constraint = E.det()

e = essential_constraint

import IPython; IPython.embed()

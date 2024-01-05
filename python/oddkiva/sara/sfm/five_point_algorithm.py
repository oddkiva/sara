def calculate_E_matrix_constraints_from_5_point():
    import sympy as sp
    from sympy.abc import x, y, z

    # Solve the cubic polynomial.
    #
    # The essential matrix lives in the nullspace which is spanned by 4
    # eigenvectors.
    X = sp.MatrixSymbol('X', 3, 3)
    Y = sp.MatrixSymbol('Y', 3, 3)
    Z = sp.MatrixSymbol('Z', 3, 3)
    W = sp.MatrixSymbol('W', 3, 3)

    # Form the symbolic matrix expression as reported in the paper.
    E = sp.Matrix(x * X + y * Y + z * Z + W)

    # Form the first system of equations that the essential matrix must
    # satisfy:
    essential_constraint = E * E.T * E - sp.Trace(E * E.T) * E

    # Form the second system of equations. The essential matrix has rank 2,
    # therefore its determinant must be zero.
    rank_2_constraint = E.det()

    return essential_constraint, rank_2_constraint

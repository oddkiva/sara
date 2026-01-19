# Copyright (C) 2025 David Ok <david.ok8@gmail.com>

def calculate_F_matrix_rank_2_constraint_from_7_points():
    import sympy as sp

    # Solve the cubic polynomial.
    F1 = sp.MatrixSymbol('F1', 3, 3)
    F2 = sp.MatrixSymbol('F2', 3, 3)
    α = sp.symbols('α')

    # The fundamental matrix has rank 2.
    # Necessarily its determinant must be 0.
    #
    # Form the symbolic matrix expression as reported in the paper.
    F = sp.Matrix(F1 + α * F2)

    # Form the polynomial in the variable α.
    det_F, _ = sp.polys.poly_from_expr(F.det(), α)

    # Collect the coefficients "c[i]" as denoted in the paper.
    c = det_F.all_coeffs()

    return c

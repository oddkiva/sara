import sympy as sp

# Solve the cubic polynomial.
D1 = sp.MatrixSymbol('D1', 3, 3)
D2 = sp.MatrixSymbol('D2', 3, 3)
gamma = sp.symbols('gamma')

# Form the symbolic matrix expression as reported in the paper.
D0 = sp.Matrix(D1 + gamma *  D2)

# Form the polynomial in the variable gamma.
det_D0, _ = sp.polys.poly_from_expr(D0.det(), gamma)

# Collect the coefficients "c[i]" as denoted in the paper.
c = det_D0.all_coeffs()

D1 = sp.Matrix(D1)
D2 = sp.Matrix(D2)


our_proof_read_coefficients = [
    D2.det(),
    (D1[:,0].T * (D2[:,1].cross(D2[:,2])) + \
     D1[:,1].T * (D2[:,2].cross(D2[:,0])) + \
     D1[:,2].T * (D2[:,0].cross(D2[:,1])))[0, 0],
    (D2[:,0].T * (D1[:,1].cross(D1[:,2])) + \
     D2[:,1].T * (D1[:,2].cross(D1[:,0])) + \
     D2[:,2].T * (D1[:,0].cross(D1[:,1])))[0, 0],
    D1.det()
]

for (c, our_coeff) in zip(c, our_proof_read_coefficients):
    print(sp.simplify(c - our_coeff))


# Auxiliary variables.
a01, a02, a12 = sp.symbols('a01 a02 a12')
b01, b02, b12 = sp.symbols('b01 b02 b12')

M01 = sp.Matrix([[   1, -b01, 0],
                 [-b01,    1, 0],
                 [   0,    0, 0]])
M02 = sp.Matrix([[   1, 0, -b02],
                 [   0, 0,    0],
                 [-b02, 0,    1]])
M12 = sp.Matrix([[0,    0,    0],
                 [0,    1, -b12],
                 [0, -b12,    1]])

D1 = M01 * a12 - M12 * a01
D2 = M02 * a12 - M12 * a02

w0, w1, tau, l = sp.symbols('w0 w1 tau l')

v = sp.Matrix([[w0 + tau * w1],
               [1],
               [tau]])

tau_polynomial = v.transpose() * D2 * v
tau_polynomial = sp.expand(tau_polynomial)[0, 0]
tau_polynomial = sp.collect(tau_polynomial, tau)
print('tau = ', tau_polynomial)

lvec = sp.Matrix([[w0 + tau * w1],
                  [1],
                  [tau]])
expr = sp.expand(lvec.transpose() * M12 * lvec);

import IPython; IPython.embed()

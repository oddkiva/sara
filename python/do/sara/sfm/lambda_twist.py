import sympy as sp

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

# # Solve the cubic polynomial.
# D1 = sp.MatrixSymbol('D1', 3, 3)
# D2 = sp.MatrixSymbol('D2', 3, 3)
# gamma = sp.symbols('gamma')
# 
# D0 = sp.Matrix(D1 + gamma * D2)
# det_D0 = D0.det()
 
# Tau polynomial.
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

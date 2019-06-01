from sympy import *

X = Matrix(symbols(' '.join(['X{}'.format(i) for i in range(9)]))).reshape(3, 3)
Y = Matrix(symbols(' '.join(['Y{}'.format(i) for i in range(9)]))).reshape(3, 3)
Z = Matrix(symbols(' '.join(['Z{}'.format(i) for i in range(9)]))).reshape(3, 3)
W = Matrix(symbols(' '.join(['W{}'.format(i) for i in range(9)]))).reshape(3, 3)

x, y, z = symbols('x y z')


E = x * X + y * Y + z * Z + W

a = det(E) 

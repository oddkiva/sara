import numpy as np

# Numpy way of calculating eigen vector
def evec_from_numpy(A):
    _, evec_A = np.linalg.eig(A)
    return evec_A

# Let's assume we have already know eval_A but not evec_A
def evec_from_eval(A, eval_A):
    n = A.shape[0]
    evec_A = np.zeros((n, n))
    for k in range(n):
        M = np.delete(A, k, axis=0)
        M = np.delete(M, k, axis=1)
        eval_M = np.linalg.eigvals(M)

        numerator = [np.prod(eval_A[i] - eval_M) for i in range(n)]
        denominator = [np.prod(np.delete(eval_A[i] - eval_A, i)) for i in range(n)]
        print("=============================")
        print(" k = ", k)
        print(" eval_M = \n", eval_M)
        print(" eval_A_0 = \n", np.delete(eval_A[0] - eval_A, 0))
        print(" eval_A_1 = \n", np.delete(eval_A[0] - eval_A, 1))
        print(" eval_A_2 = \n", np.delete(eval_A[0] - eval_A, 2))
        print(" num =\n", numerator)
        print(" den =\n", denominator)
        print()

        evec_A[k, :] = np.array(numerator) / np.array(denominator)
    return evec_A

n = 3
A = np.array([[   0.675, 0.736122, 0],
              [0.736122,   -0.175, 0],
              [       0,        0, 0]])
eval_A, evec_A = np.linalg.eig(A)
print("eval_A =\n", eval_A)
evec_A_from_eval = np.zeros((n, n))
evec_A_from_eval = evec_from_eval(A, eval_A)

print(evec_A_from_eval)
print(evec_A ** 2)

print(np.allclose(evec_A_from_eval, evec_A**2))

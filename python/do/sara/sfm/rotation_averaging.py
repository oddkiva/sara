import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


def rotation_averaging_chatterjee_govindu_l1ra_init(V, E, R_rel, R_abs, eps):
    pass

def rotation_averaging_chatterjee_govindu_l1ra(V, E, R_rel, R_abs, eps):
    card_V = len(V)
    card_E = len(E)

    delta_R_rel_dict = {}
    delta_w_rel_dict = {}

    R_rel = sp.dok_matrix(3 * card_V * 3 * card_V)
    delta_R_rel = sp.dok_matrix(3 * card_V * 3 * card_V)
    delta_w_rel = sp.dok_matrix(3 * card_V * 3 * card_V)

    norm_delta_w_rel = np.inf

    while norm_delta_w_rel > eps:
        # Step 1.
        for (i, j) in E:
            delta_R_rel[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = \
                R_abs[j].T.dot(R_rel[(i, j)]).dot(R_abs[i])

        # Step 2.
        for (i, j) in E:
            delta_w_rel[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = \
                la.logm(delta_R_rel[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])

        # Step 3.
        # TODO: use the L1 norm instead, use the ADMM method.
        delta_w_abs = la.lstsq(A, delta_w_rel)

        # Step 4.
        R_abs[i * 3:(i + 1) * 3] = R_abs[i].dot(expm(delta_abs[i]))


def rotation_averaging_chatterjee_govindu_irls(V, E, R_rel, R_abs, phi, eps):
    card_V = len(V)
    card_E = len(E)

    delta_R_rel_dict = {}
    delta_w_rel_dict = {}

    R_rel = sp.dok_matrix(3 * card_V * 3 * card_V)
    delta_R_rel = sp.dok_matrix(3 * card_V * 3 * card_V)
    delta_w_rel = sp.dok_matrix(3 * card_V * 3 * card_V)

    norm_delta_w_rel = np.inf

    while la.norm(w_abs_1- w_abs_0) > eps:
        # Step 1.
        w_abs_0 = w_abs_1

        # Step 2.
        e = A.dot(w_abs_1) - w_rel

        # Step 3.
        phi = phi(e)

        # Step 4.
        w_abs_1 = la.lstsq(A.T.dot(phi).dot(A), A.dot(phi).dot(w_rel))

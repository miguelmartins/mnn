import math
import numpy as np


def simplex_projection_1d(vec):
    """
    Given a vector project it onto to the canonical simplex polyhedron in R^n

    Based on:
    Michelot, C., 1986. A finite algorithm for finding the projection of a point onto the canonical simplex of∝^n.
    Journal of Optimization Theory and Applications, 50(1), pp.195-200.

    Parameters
    ----------
    optimizer : numpy.ndarray
       A vector in R^N

    Returns
    -------
    numpy.ndarray
        The projection of the input vector onto the canonical simplex of R^n
    """
    if len(vec.shape) == 1:
        xt = vec.reshape((vec.shape[0], 1))
    else:
        xt = vec
    n = len(vec)
    In = np.ones((n, 1))
    for k in range(n):
        I = np.where(In == 0)[0]
        Ineg = np.where(In == 1)[0]
        dimIneg = len(Ineg)

        # project onto hyperplane
        x_til = np.zeros((n, 1))
        x_til[I] = 0
        x_til[Ineg] = xt[Ineg] - \
                         (np.matmul(np.ones((dimIneg, dimIneg)), xt[Ineg]) - np.ones((dimIneg, 1))) / dimIneg
        zero_idx = np.where(x_til < 0)
        if len(zero_idx) == 0:
            return x_til
        else:
            In[zero_idx] = 0
            xt = x_til
            xt[zero_idx] = 0
    return x_til


def projection_simplex(V, z=1, axis=None):
    """
    Source: https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


def project_matrix_row_components(trans_mat, jitter_treshold=0.005, jitter=0.1):
    """

    Parameters
    ----------
    trans_mat A left to right markovian matrix

    Returns The canonical simplex projected only on the positive components
    -------

    """
    num_zeros = trans_mat.shape[0] - 2
    for offset, row in enumerate(trans_mat):
        proj = projection_simplex(np.roll(row, -offset)[:2])
        # Add a jitter when the row is no longer stochastic
        if np.any([proj < jitter_treshold]):
            idx = np.where(proj < jitter_treshold)[0]
            idx_1 = (idx + 1) % 2
            proj[idx], proj[idx_1] = jitter, 1 - jitter
        prob_vector = np.concatenate([proj, np.zeros(num_zeros)])
        proj = np.roll(prob_vector, offset)
        trans_mat[offset, :] = proj
    return trans_mat


def simplex_projection(mat):
    """
    Calls a canonical simplex polyhedron projection for R^n for the the rowspace of the input Matrix mat

    Parameters
    ----------
    optimizer : numpy.ndarray
       A matrix in R^{M*N}

    Returns
    -------
    numpy.ndarray
        The projection of the rowspace onto the canonical simplex of R^n
    """
    if len(mat.shape) == 1:
        projection = simplex_projection_1d(mat)
        return projection.reshape((projection.shape[0],))

    n_rows = mat.shape[0]
    for i in range(n_rows):
        projection = simplex_projection_1d(mat[i])
        try:
            mat[i, :] = projection
        except:
            mat[i, :] = projection.reshape((projection.shape[0],))
    return mat

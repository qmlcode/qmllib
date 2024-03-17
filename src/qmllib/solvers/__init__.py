from typing import Optional

import numpy as np
from numpy import ndarray

from .fsolvers import (
    fbkf_invert,
    fbkf_solve,
    fcho_invert,
    fcho_solve,
    fcond,
    fcond_ge,
    fqrlq_solve,
    fsvd_solve,
)


def cho_invert(A: ndarray) -> ndarray:
    """Returns the inverse of a positive definite matrix, using a Cholesky decomposition
    via calls to LAPACK dpotrf and dpotri in the F2PY module.

    :param A: Matrix (symmetric and positive definite, left-hand side).
    :type A: numpy array

    :return: The inverse matrix
    :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")

    matrix = np.asfortranarray(A)

    fcho_invert(matrix)

    # Matrix to store the inverse
    i_lower = np.tril_indices_from(A)

    # Copy lower triangle to upper
    matrix.T[i_lower] = matrix[i_lower]

    return matrix


def cho_solve(A: ndarray, y: ndarray, l2reg: float = 0.0, destructive: bool = False) -> ndarray:
    """Solves the equation

        :math:`A x = y`

    for x using a Cholesky decomposition  via calls to LAPACK dpotrf and dpotrs in the F2PY module. Preserves the input matrix A.

    :param A: Matrix (symmetric and positive definite, left-hand side).
    :type A: numpy array
    :param y: Vector (right-hand side of the equation).
    :type y: numpy array
    :param l2reg: Small number to add to the diagonal as L2-regularization when solving.
    :type l2reg: float
    :param destructive: Whether to preserve the lower triangle after solving(=False) or destroy it, which is faster(=True).
    :type destructive: bool


    :return: The solution vector.
    :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")

    if len(y.shape) != 1 or y.shape[0] != A.shape[1]:
        raise ValueError("expected matrix and vector of same stride size")

    n = A.shape[0]

    # Backup diagonal before Cholesky-decomposition
    A_diag = A[np.diag_indices_from(A)]

    for i in range(len(y)):

        A[i, i] += l2reg

    x = np.zeros(n)
    fcho_solve(A, y, x)

    # Reset diagonal after Cholesky-decomposition
    A[np.diag_indices_from(A)] = A_diag

    if destructive is False:

        # Copy lower triangle to upper
        i_lower = np.tril_indices_from(A)
        A.T[i_lower] = A[i_lower]

    return x


def bkf_invert(A: ndarray) -> ndarray:
    """Returns the inverse of a positive definite matrix, using a Bausch-Kauffman decomposition
    via calls to LAPACK dpotrf and dpotri in the F2PY module.

    :param A: Matrix (symmetric and positive definite, left-hand side).
    :type A: numpy array

    :return: The inverse matrix
    :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")

    matrix = np.asfortranarray(A)

    fbkf_invert(matrix)

    # Matrix to store the inverse
    i_lower = np.tril_indices_from(A)

    # Copy lower triangle to upper
    matrix.T[i_lower] = matrix[i_lower]

    return matrix


def bkf_solve(A: ndarray, y: ndarray) -> ndarray:
    """Solves the equation

        :math:`A x = y`

    for x using a  Bausch-Kauffma  decomposition  via calls to LAPACK  in the F2PY module. Preserves the input matrix A.

    :param A: Matrix (symmetric and positive definite, left-hand side).
    :type A: numpy array
    :param y: Vector (right-hand side of the equation).
    :type y: numpy array

    :return: The solution vector.
    :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")

    if len(y.shape) != 1 or y.shape[0] != A.shape[1]:
        raise ValueError("expected matrix and vector of same stride size")

    n = A.shape[0]

    # Backup diagonal before Cholesky-decomposition
    A_diag = A[np.diag_indices_from(A)]

    x = np.zeros(n)
    fbkf_solve(A, y, x)

    # Reset diagonal after Cholesky-decomposition
    A[np.diag_indices_from(A)] = A_diag

    # Copy lower triangle to upper
    i_lower = np.tril_indices_from(A)
    A.T[i_lower] = A[i_lower]

    return x


def svd_solve(A: ndarray, y: ndarray, rcond: Optional[float] = None) -> ndarray:
    """Solves the equation

        :math:`A x = y`

    for x using a singular-value decomposition (SVD) via calls to
    LAPACK DGELSD in the F2PY module. Preserves the input matrix A.

    :param A: Matrix (symmetric and positive definite, left-hand side).
    :type A: numpy array
    :param y: Vector (right-hand side of the equation).
    :type y: numpy array
    :param rcond: Optional parameater for lowest singular-value
    :type rcond: float

    :return: The solution vector.
    :rtype: numpy array
    """

    if len(y.shape) != 1 or y.shape[0] != A.shape[0]:
        raise ValueError("expected matrix and vector of same stride size")

    if rcond is None:
        rcond = 0.0

    x_dim = A.shape[1]
    A = np.asarray(A, order="F")
    x = fsvd_solve(A, y, x_dim, rcond)

    return x


def qrlq_solve(A, y):
    """Solves the equation

        :math:`A x = y`

    for x using a QR or LQ decomposition (depending on matrix dimensions)
    via calls to LAPACK DGELSD in the F2PY module. Preserves the input matrix A.

    :param A: Matrix (symmetric and positive definite, left-hand side).
    :type A: numpy array
    :param y: Vector (right-hand side of the equation).
    :type y: numpy array

    :return: The solution vector.
    :rtype: numpy array
    """

    if len(y.shape) != 1 or y.shape[0] != A.shape[0]:
        raise ValueError("expected matrix and vector of same stride size")

    x_dim = A.shape[1]
    A = np.asarray(A, order="F")
    x = fqrlq_solve(A, y, x_dim)

    return x


def condition_number(A, method="cholesky"):
    """Returns the condition number for the square matrix A.

    Two different methods are implemented:
    Cholesky (requires a positive-definite matrix), but barely any additional memory overhead.
    LU: Does not require a positive definite matrix, but requires additional memory.
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")

    if method.lower() == "cholesky":
        assert np.allclose(
            A, A.T
        ), "ERROR: Can't use a Cholesky-decomposition for a non-symmetric matrix."

        cond = fcond(A)

        return cond

    elif method.lower() == "lu":

        cond = fcond_ge(A)

        return cond


#

#






#


#








import numpy as np

from copy import deepcopy

from .fsolvers import fcho_solve
from .fsolvers import fcho_invert
from .fsolvers import fbkf_solve
from .fsolvers import fbkf_invert


def cho_invert(A):
    """ Returns the inverse of a positive definite matrix, using a Cholesky decomposition
        via calls to LAPACK dpotrf and dpotri in the F2PY module.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array

        :return: The inverse matrix
        :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    I = np.asfortranarray(A)

    fcho_invert(I)

    # Matrix to store the inverse
    i_lower = np.tril_indices_from(A)

    # Copy lower triangle to upper
    I.T[i_lower] = I[i_lower]

    return I


def cho_solve(A, y):
    """ Solves the equation

            :math:`A x = y`

        for x using a Cholesky decomposition  via calls to LAPACK dpotrf and dpotrs in the F2PY module. Preserves the input matrix A.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array
        :param y: Vector (right-hand side of the equation).
        :type y: numpy array

        :return: The solution vector.
        :rtype: numpy array
        """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if len(y.shape) != 1 or y.shape[0] != A.shape[1]:
        raise ValueError('expected matrix and vector of same stride size')

    n = A.shape[0]

    # Backup diagonal before Cholesky-decomposition
    A_diag = A[np.diag_indices_from(A)]

    x = np.zeros(n)
    fcho_solve(A, y, x)

    # Reset diagonal after Cholesky-decomposition
    A[np.diag_indices_from(A)] = A_diag

    # Copy lower triangle to upper
    i_lower = np.tril_indices_from(A)
    A.T[i_lower] = A[i_lower]

    return x


def bkf_invert(A):
    """ Returns the inverse of a positive definite matrix, using a Cholesky decomposition
        via calls to LAPACK dpotrf and dpotri in the F2PY module.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array

        :return: The inverse matrix
        :rtype: numpy array
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    I = np.asfortranarray(A)

    fbkf_invert(I)

    # Matrix to store the inverse
    i_lower = np.tril_indices_from(A)

    # Copy lower triangle to upper
    I.T[i_lower] = I[i_lower]

    return I


def bkf_solve(A, y):
    """ Solves the equation

            :math:`A x = y`

        for x using a Cholesky decomposition  via calls to LAPACK dpotrf and dpotrs in the F2PY module. Preserves the input matrix A.

        :param A: Matrix (symmetric and positive definite, left-hand side).
        :type A: numpy array
        :param y: Vector (right-hand side of the equation).
        :type y: numpy array

        :return: The solution vector.
        :rtype: numpy array
        """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if len(y.shape) != 1 or y.shape[0] != A.shape[1]:
        raise ValueError('expected matrix and vector of same stride size')

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

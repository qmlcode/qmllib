
#

#






#


#








import numpy as np

from fcho_solve import fcho_solve
from fcho_solve import fcho_invert

def cho_invert(A):
    """ Solves [A x = y] for x using a Cholesky decomposition
        via calls to LAPACK dpotrf and dpotri in the F2PY module.

        Arguments:
        ==============
        A -- the A-matrix (symmetric and positive definite).

        Returns:
        ==============
        A -- the inverted A-matrix
    """

    A = np.asfortranarray(A)
    fcho_invert(A)

    return A


def cho_solve(A, y):
    """ Solves [A x = y] for x using a Cholesky decomposition
        via calls to LAPACK dpotrf and dpotrs in the F2PY module.

        Arguments:
        ==============
        A -- the A-matrix (symmetric and positive definite).
        y -- the right-hand side of the equation (vector).

        Returns:
        ==============
        x -- the vector for with the equation has been solved.
    """

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    if len(y.shape) != 1 or y.shape[0] != A.shape[1]:
        raise ValueError('expected matrix and vector of same stride size')

    n = A.shape[0]

    x = np.zeros((n))
    fcho_solve(A,y,x)

    return x

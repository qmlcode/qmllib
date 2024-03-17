from typing import Union

import numpy as np
from numpy import ndarray

from .fdistance import fl2_distance, fmanhattan_distance, fp_distance_double, fp_distance_integer


def manhattan_distance(A: ndarray, B: ndarray) -> ndarray:
    """Calculates the Manhattan distances, D,  between two
    Numpy arrays of representations.

        :math:`D_{ij} = \\|A_i - B_j\\|_1`

    Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
    D is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of descriptors - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of descriptors - shape (M, representation size).
    :type B: numpy array

    :return: The Manhattan-distance matrix.
    :rtype: numpy array
    """

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("expected matrices of dimension=2")

    if B.shape[1] != A.shape[1]:
        raise ValueError("expected matrices containing vectors of same size")

    na = A.shape[0]
    nb = B.shape[0]

    D = np.empty((na, nb), order="F")

    fmanhattan_distance(A.T, B.T, D)

    return D


def l2_distance(A: ndarray, B: ndarray) -> ndarray:
    """Calculates the L2 distances, D, between two
    Numpy arrays of representations.

        :math:`D_{ij} = \\|A_i - B_j\\|_2`

    Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
    D is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of descriptors - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of descriptors - shape (M, representation size).
    :type B: numpy array

    :return: The L2-distance matrix.
    :rtype: numpy array
    """

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("expected matrices of dimension=2")

    if B.shape[1] != A.shape[1]:
        raise ValueError("expected matrices containing vectors of same size")

    na = A.shape[0]
    nb = B.shape[0]

    D = np.empty((na, nb), order="F")

    fl2_distance(A.T, B.T, D)

    return D


def p_distance(A: ndarray, B: ndarray, p: Union[int, float] = 2) -> ndarray:
    """Calculates the p-norm distances between two
    Numpy arrays of representations.
    The value of the keyword argument ``p =`` sets the norm order.
    E.g. ``p = 1.0`` and ``p = 2.0`` with yield the Manhattan and L2 distances, respectively.

        .. math:: D_{ij} = \\|A_i - B_j\\|_p

    Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
    D is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of descriptors - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of descriptors - shape (M, representation size).
    :type B: numpy array
    :param p: The norm order
    :type p: float

    :return: The distance matrix.
    :rtype: numpy array
    """

    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("expected matrices of dimension=2")

    if B.shape[1] != A.shape[1]:
        raise ValueError("expected matrices containing vectors of same size")

    na = A.shape[0]
    nb = B.shape[0]

    D = np.empty((na, nb), order="F")

    if isinstance(p, int):
        if p == 2:
            fl2_distance(A, B, D)
        else:
            fp_distance_integer(A.T, B.T, D, p)

    elif isinstance(p, float):
        if p.is_integer():
            p = int(p)
            if p == 2:
                fl2_distance(A, B, D)
            else:
                fp_distance_integer(A.T, B.T, D, p)

        else:
            fp_distance_double(A.T, B.T, D, p)
    else:
        raise ValueError("expected exponent of integer or float type")

    return D

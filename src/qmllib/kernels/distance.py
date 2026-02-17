from typing import Union

import numpy as np
from numpy import ndarray

# Import from pybind11 module
from qmllib._fdistance import (
    fl2_distance,
    fmanhattan_distance,
    fp_distance_double,
    fp_distance_integer,
)


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

    # Call the pybind11 function which returns the result
    D = fmanhattan_distance(A.T, B.T)

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

    # Call the pybind11 function which returns the result
    D = fl2_distance(A.T, B.T)

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

    # Call the pybind11 function which returns the result
    if isinstance(p, int):
        if p == 2:
            D = fl2_distance(A.T, B.T)
        else:
            D = fp_distance_integer(A.T, B.T, p)

    elif isinstance(p, float):
        if p.is_integer():
            p = int(p)
            if p == 2:
                D = fl2_distance(A.T, B.T)
            else:
                D = fp_distance_integer(A.T, B.T, p)

        else:
            D = fp_distance_double(A.T, B.T, p)
    else:
        raise ValueError("expected exponent of integer or float type")

    return D

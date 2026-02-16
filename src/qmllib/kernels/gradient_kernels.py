from typing import List, Union

import numpy as np
from numpy import ndarray

from qmllib.utils.environment_manipulation import (
    mkl_get_num_threads,
    mkl_reset_num_threads,
    mkl_set_num_threads,
)

# Import from pybind11 module
from qmllib._fgradient_kernels import (
    fatomic_local_gradient_kernel,
    fatomic_local_kernel,
    fgaussian_process_kernel,
    fgdml_kernel,
    fglobal_kernel,
    flocal_gradient_kernel,
    flocal_kernel,
    flocal_kernels,
    fsymmetric_gaussian_process_kernel,
    fsymmetric_gdml_kernel,
    fsymmetric_local_kernel,
    fsymmetric_local_kernels,
)


def get_global_kernel(
    X1: ndarray, X2: ndarray, Q1: List[List[int]], Q2: List[List[int]], SIGMA: float
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{ij} = \\sum_{I\\in i} \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.

    The kernel is the normal Gaussian kernel with the local decomposition of atomic environments.

    For instance atom-centered symmetry functions could be used here.
    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError(
            "Error: List of charges does not match shape of representations"
        )

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    K = fglobal_kernel(X1, X2, Q1_input, Q2_input, N1, N2, len(N1), len(N2), SIGMA)

    return K


def get_local_kernels(
    X1: ndarray,
    X2: ndarray,
    Q1: List[List[int]],
    Q2: List[List[int]],
    SIGMAS: List[float],
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{ij} = \\sum_{I\\in i} \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.

    The kernel is the normal Gaussian kernel with the local decomposition of atomic environments.

    For instance atom-centered symmetry functions could be used here.
    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError(
            "Error: List of charges does not match shape of representations"
        )
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError(
            "Error: List of charges does not match shape of representations"
        )

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    sigmas_input = np.array(SIGMAS, dtype=np.float64)
    nsigmas = len(SIGMAS)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    K = flocal_kernels(
        X1, X2, Q1_input, Q2_input, N1, N2, len(N1), len(N2), sigmas_input, nsigmas
    )

    return K


def get_local_kernel(
    X1: ndarray,
    X2: ndarray,
    Q1: List[Union[ndarray, List[int]]],
    Q2: List[Union[ndarray, List[int]]],
    SIGMA: float,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{ij} = \\sum_{I\\in i} \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.

    The kernel is the normal Gaussian kernel with the local decomposition of atomic environments.

    For instance atom-centered symmetry functions could be used here.
    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    # CRITICAL: Q_input arrays must match X's padding size (X.shape[1]), not just max(N)
    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    # Convert to Fortran order for compatibility with Fortran routine
    X1_f = np.asfortranarray(X1)
    X2_f = np.asfortranarray(X2)
    Q1_input_f = np.asfortranarray(Q1_input)
    Q2_input_f = np.asfortranarray(Q2_input)
    N1_f = np.asfortranarray(N1)
    N2_f = np.asfortranarray(N2)

    K = flocal_kernel(
        X1_f, X2_f, Q1_input_f, Q2_input_f, N1_f, N2_f, len(N1), len(N2), SIGMA
    )

    return K


def get_local_symmetric_kernels(
    X1: ndarray, Q1: List[List[int]], SIGMAS: List[float]
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{ij} = \\sum_{I\\in i} \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.

    The kernel is the normal Gaussian kernel with the local decomposition of atomic environments.

    For instance atom-centered symmetry functions could be used here.
    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError(
            "Error: List of charges does not match shape of representations"
        )

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    nsigmas = len(SIGMAS)
    K = fsymmetric_local_kernels(X1, Q1_input, N1, len(N1), SIGMAS, nsigmas)

    return K


def get_local_symmetric_kernel(
    X1: ndarray, Q1: List[Union[ndarray, List[int]]], SIGMA: float
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{ij} = \\sum_{I\\in i} \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.

    The kernel is the normal Gaussian kernel with the local decomposition of atomic environments.

    For instance atom-centered symmetry functions could be used here.
    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError(
            "Error: List of charges does not match shape of representations"
        )

    # CRITICAL: Q1_input must match X1's padding size (X1.shape[1]), not just max(N1)
    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    # Convert to Fortran order for compatibility with Fortran routine
    X1_f = np.asfortranarray(X1)
    Q1_input_f = np.asfortranarray(Q1_input)
    N1_f = np.asfortranarray(N1)

    K = fsymmetric_local_kernel(X1_f, Q1_input_f, N1_f, len(N1), SIGMA)

    return K


def get_atomic_local_kernel(
    X1: ndarray,
    X2: ndarray,
    Q1: List[Union[ndarray, List[int]]],
    Q2: List[Union[ndarray, List[int]]],
    SIGMA: float,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{Ij} = \\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.

    This means that the kernel matrix consists of the local decomposition of atomic environments
    in a basis of kernel function placed on single atomic environments.

    Thus the dimensions in number of molecules times total number of atoms.

    For instance atom-centered symmetry functions could be used here.
    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    K = fatomic_local_kernel(
        X1, X2, Q1_input, Q2_input, N1, N2, len(N1), len(N2), np.sum(N1), SIGMA
    )

    return K


def get_atomic_local_gradient_kernel(
    X1: ndarray,
    X2: ndarray,
    dX2: ndarray,
    Q1: List[Union[ndarray, List[int]]],
    Q2: List[Union[ndarray, List[int]]],
    SIGMA: float,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{Ij} = \\frac{\\part}{\\part x}\\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.
    For instance atom-centered symmetry functions could be used here.

    The kernel has the dimensions number of nuclear kernel gradients times total number of atoms.

    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param dX2: Array of representation derivatives - shape=(N2, rep_size, 3, rep_size, max_atoms).
    :type dX2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = mkl_get_num_threads()
    mkl_set_num_threads(1)

    K = fatomic_local_gradient_kernel(
        X1,
        X2,
        dX2,
        Q1_input,
        Q2_input,
        N1,
        N2,
        len(N1),
        len(N2),
        np.sum(N1),
        np.sum(N2) * 3,
        SIGMA,
    )

    # Reset MKL_NUM_THREADS back to its original value
    if original_mkl_threads is not None:
        mkl_set_num_threads(original_mkl_threads)
    else:
        mkl_reset_num_threads()

    return K


def get_local_gradient_kernel(
    X1: ndarray,
    X2: ndarray,
    dX2: ndarray,
    Q1: List[List[int]],
    Q2: List[List[int]],
    SIGMA: float,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{ij} = \\frac{\\part}{\\part x}\\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.
    For instance atom-centered symmetry functions could be used here.

    The kernel has the dimensions number of nuclear kernel gradients times total number of molecules.

    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param dX2: Array of representation derivatives - shape=(N2, rep_size, 3, rep_size, max_atoms).
    :type dX2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = mkl_get_num_threads()
    mkl_set_num_threads(1)

    K = flocal_gradient_kernel(
        X1, X2, dX2, Q1_input, Q2_input, N1, N2, len(N1), len(N2), np.sum(N2) * 3, SIGMA
    )

    # Reset MKL_NUM_THREADS back to its original value
    mkl_set_num_threads(original_mkl_threads)

    return K


def get_gdml_kernel(
    X1: ndarray,
    X2: ndarray,
    dX1: ndarray,
    dX2: ndarray,
    Q1: List[List[int]],
    Q2: List[List[int]],
    SIGMA: float,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{Ij} = \\frac{\\part^2}{\\part x_i\\part x_j}\\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.
    For instance atom-centered symmetry functions could be used here.

    This Hessian-kernel corresponds to the "gradient-domain machine learning" (GDML) approach.
    This means that the surface is only trained on its derivatives.

    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param dX1: Array of representation derivatives - shape=(N1, rep_size, 3, rep_size, max_atoms).
    :type dX1: numpy array

    :param dX2: Array of representation derivatives - shape=(N2, rep_size, 3, rep_size, max_atoms).
    :type dX2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = mkl_get_num_threads()
    mkl_set_num_threads(1)

    K = fgdml_kernel(
        X1,
        X2,
        dX1,
        dX2,
        Q1_input,
        Q2_input,
        N1,
        N2,
        len(N1),
        len(N2),
        np.sum(N1),
        np.sum(N2),
        SIGMA,
    )

    # Reset MKL_NUM_THREADS back to its original value
    mkl_set_num_threads(original_mkl_threads)

    return K


def get_symmetric_gdml_kernel(
    X1: ndarray, dX1: ndarray, Q1: List[List[int]], SIGMA: float
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

        :math:`K_{Ij} = \\frac{\\part^2}{\\part x_i\\part x_j}\\sum_{J\\in j}\\exp \\big( -\\frac{\\|X_I - X_J\\|_2^2}{2\\sigma^2} \\big)`

    Where :math: X_{I}` and :math:`X_{J}` are representation vectors of the atomic environments.
    For instance atom-centered symmetry functions could be used here.

    This symmetric Hessian-kernel corresponds to the "gradient-domain machine learning" (GDML) approach.
    This means that the surface is only trained on its derivatives.

    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array

    :param dX1: Array of representation derivatives - shape=(N1, rep_size, 3, rep_size, max_atoms).
    :type dX1: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = mkl_get_num_threads()
    mkl_set_num_threads(1)

    K = fsymmetric_gdml_kernel(X1, dX1, Q1_input, N1, len(N1), np.sum(N1), SIGMA)

    # Reset MKL_NUM_THREADS back to its original value
    mkl_set_num_threads(original_mkl_threads)

    return K


def get_gp_kernel(
    X1: ndarray,
    X2: ndarray,
    dX1: ndarray,
    dX2: ndarray,
    Q1: List[Union[ndarray, List[int]]],
    Q2: List[Union[ndarray, List[int]]],
    SIGMA: float,
) -> ndarray:
    """Calculates the Gaussian kernel matrix K with the local decomposition where :math:`K_{ij}`:

    This kernel corresponds to a Gaussian process regression (GPR) approach.
    The kernel has four blocks, consisting of the 0'th, 1st and 2nd derivatives.

    The size is (the number of gradients plus the number of molecules) squared.

    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array
    :param X2: Array of representations - shape=(N2, rep_size, max_atoms).
    :type X2: numpy array

    :param dX1: Array of representation derivatives - shape=(N1, rep_size, 3, rep_size, max_atoms).
    :type dX1: numpy array

    :param dX2: Array of representation derivatives - shape=(N2, rep_size, 3, rep_size, max_atoms).
    :type dX2: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)
    N2 = np.array([len(Q) for Q in Q2], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")
    if not (N2.shape[0] == X2.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)
    Q2_input = np.zeros((X2.shape[1], X2.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    for i, q in enumerate(Q2):
        Q2_input[: len(q), i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = mkl_get_num_threads()
    mkl_set_num_threads(1)

    K = fgaussian_process_kernel(
        X1,
        X2,
        dX1,
        dX2,
        Q1_input,
        Q2_input,
        N1,
        N2,
        len(N1),
        len(N2),
        np.sum(N1),
        np.sum(N2),
        SIGMA,
    )

    # Reset MKL_NUM_THREADS back to its original value
    mkl_set_num_threads(original_mkl_threads)

    return K


def get_symmetric_gp_kernel(
    X1: ndarray, dX1: ndarray, Q1: List[Union[ndarray, List[int]]], SIGMA: float
) -> ndarray:
    """
    This symmetric kernel corresponds to a Gaussian process regression (GPR) approach.
    The kernel has four blocks, consisting of the 0'th, 1st and 2nd derivatives.

    The size is (the number of gradients plus the number of molecules) squared.

    K is calculated analytically using an OpenMP parallel Fortran routine.

    :param X1: Array of representations - shape=(N1, rep_size, max_atoms).
    :type X1: numpy array

    :param dX1: Array of representation derivatives - shape=(N1, rep_size, 3, rep_size, max_atoms).
    :type dX1: numpy array

    :param Q1: List of lists containing the nuclear charges for each molecule.
    :type Q1: list
    :param Q2: List of lists containing the nuclear charges for each molecule.
    :type Q2: list

    :param SIGMA: Gaussian kernel width.
    :type SIGMA: float

    :return: 2D matrix of kernel elements shape=(N1, N2),
    :rtype: numpy array
    """

    N1 = np.array([len(Q) for Q in Q1], dtype=np.int32)

    if not (N1.shape[0] == X1.shape[0]):
        raise ValueError("List of charges does not match shape of representations")

    Q1_input = np.zeros((X1.shape[1], X1.shape[0]), dtype=np.int32)

    for i, q in enumerate(Q1):
        Q1_input[: len(q), i] = q

    # This kernel must run with MKL_NUM_THREADS=1
    original_mkl_threads = mkl_get_num_threads()
    mkl_set_num_threads(1)

    K = fsymmetric_gaussian_process_kernel(
        X1, dX1, Q1_input, N1, len(N1), np.sum(N1), SIGMA
    )

    # Reset MKL_NUM_THREADS back to its original value
    mkl_set_num_threads(original_mkl_threads)

    return K

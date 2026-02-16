from typing import List, Union

import numpy as np
from numpy import float64, ndarray

# Import from pybind11 modules
from qmllib._fkernels import (
    fgaussian_kernel,
    fgaussian_kernel_symmetric,
    fget_local_kernels_gaussian,
    fget_local_kernels_laplacian,
    fkpca,
    flaplacian_kernel,
    flaplacian_kernel_symmetric,
    flinear_kernel,
    fmatern_kernel_l2,
    fsargan_kernel,
    fwasserstein_kernel,
)


def wasserstein_kernel(
    A: ndarray, B: ndarray, sigma: float, p: int = 1, q: int = 1
) -> ndarray:
    """Calculates the Wasserstein kernel matrix K, where :math:`K_{ij}`:

    :math:`K_{ij} = \\exp \\big( -\\frac{(W_p(A_i, B_i))^q}{\\sigma} \\big)`

    Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of representations - shape (M, representation size).
    :type B: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float

    :return: The Wasserstein kernel matrix - shape (N, M)
    :rtype: numpy array
    """

    na = A.shape[0]
    nb = B.shape[0]

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = fwasserstein_kernel(
        np.asfortranarray(A.T), na, np.asfortranarray(B.T), nb, sigma, p, q
    )

    return K


def laplacian_kernel(A: ndarray, B: ndarray, sigma: float) -> ndarray:
    """Calculates the Laplacian kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_1}{\\sigma} \\big)`

    Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of representations - shape (M, representation size).
    :type B: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float

    :return: The Laplacian kernel matrix - shape (N, M)
    :rtype: numpy array
    """

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = flaplacian_kernel(np.asfortranarray(A.T), np.asfortranarray(B.T), sigma)

    return K


def laplacian_kernel_symmetric(A: ndarray, sigma: float) -> ndarray:
    """Calculates the symmetric Laplacian kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - A_j\\|_1}{\\sigma} \\big)`

    Where :math:`A_{i}` are representation vectors.
    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float

    :return: The Laplacian kernel matrix - shape (N, N)
    :rtype: numpy array
    """

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = flaplacian_kernel_symmetric(np.asfortranarray(A.T), sigma)

    return K


def gaussian_kernel(A: ndarray, B: ndarray, sigma: float) -> ndarray:
    """Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\\sigma^2} \\big)`

    Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of representations - shape (M, representation size).
    :type B: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float

    :return: The Gaussian kernel matrix - shape (N, M)
    :rtype: numpy array
    """

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = fgaussian_kernel(np.asfortranarray(A.T), np.asfortranarray(B.T), sigma)

    return K


def gaussian_kernel_symmetric(A: ndarray, sigma: float) -> ndarray:
    """Calculates the symmetric Gaussian kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - A_j\\|_2^2}{2\\sigma^2} \\big)`

    Where :math:`A_{i}` are representation vectors.
    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float

    :return: The Gaussian kernel matrix - shape (N, N)
    :rtype: numpy array
    """

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = fgaussian_kernel_symmetric(np.asfortranarray(A.T), sigma)

    return K


def linear_kernel(A: ndarray, B: ndarray) -> ndarray:
    """Calculates the linear kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = A_i \\cdot B_j`

    VWhere :math:`A_{i}` and :math:`B_{j}` are  representation vectors.

    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of representations - shape (M, representation size).
    :type B: numpy array

    :return: The Gaussian kernel matrix - shape (N, M)
    :rtype: numpy array
    """

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = flinear_kernel(np.asfortranarray(A.T), np.asfortranarray(B.T))

    return K


def sargan_kernel(
    A: ndarray,
    B: ndarray,
    sigma: Union[float, float64],
    gammas: Union[ndarray, List[Union[int, float]], List[int]],
) -> ndarray:
    """Calculates the Sargan kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = \\exp \\big( -\\frac{\\| A_i - B_j \\|_1)}{\\sigma} \\big) \\big(1 + \\sum_{k} \\frac{\\gamma_{k} \\| A_i - B_j \\|_1^k}{\\sigma^k} \\big)`

    Where :math:`A_{i}` and :math:`B_{j}` are representation vectors.
    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of representations - shape (M, representation size).
    :type B: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float
    :param gammas: 1D array of parameters in the kernel matrix.
    :type gammas: numpy array

    :return: The Sargan kernel matrix - shape (N, M).
    :rtype: numpy array
    """

    ng = len(gammas)

    if ng == 0:
        return laplacian_kernel(A, B, sigma)

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = fsargan_kernel(
        np.asfortranarray(A.T), np.asfortranarray(B.T), sigma, np.asfortranarray(gammas)
    )

    return K


def matern_kernel(
    A: ndarray, B: ndarray, sigma: float, order: int = 0, metric: str = "l1"
) -> ndarray:
    """Calculates the Matern kernel matrix K, where :math:`K_{ij}`:

        for order = 0:
            :math:`K_{ij} = \\exp\\big( -\\frac{d}{\\sigma} \\big)`
        for order = 1:
            :math:`K_{ij} = \\exp\\big( -\\frac{\\sqrt{3} d}{\\sigma} \\big) \\big(1 + \\frac{\\sqrt{3} d}{\\sigma} \\big)`
        for order = 2:
            :math:`K_{ij} = \\exp\\big( -\\frac{\\sqrt{5} d}{d} \\big) \\big( 1 + \\frac{\\sqrt{5} d}{\\sigma} + \\frac{5 d^2}{3\\sigma^2} \\big)`

    Where :math:`A_i` and :math:`B_j` are representation vectors, and d is a distance measure.

    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of representations - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of representations - shape (M, representation size).
    :type B: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float
    :param order: The order of the polynomial (0, 1, 2)
    :type order: integer
    :param metric: The distance metric ('l1', 'l2')
    :type metric: string

    :return: The Matern kernel matrix - shape (N, M)
    :rtype: numpy array
    """

    if metric == "l1":
        if order == 0:
            gammas = []

        elif order == 1:
            gammas = [1]
            sigma /= np.sqrt(3)

        elif order == 2:
            gammas = [1, 1 / 3.0]
            sigma /= np.sqrt(5)

        else:
            raise ValueError(f"Order '{order}' not implemented in Matern Kernel")

        return sargan_kernel(A, B, sigma, gammas)

    elif metric == "l2":
        pass

    else:
        raise ValueError(f"Unknown distance metric {metric} in Matern kernel")

    # Transpose for Fortran column-major format (rep_size, n_samples)
    K = fmatern_kernel_l2(np.asfortranarray(A.T), np.asfortranarray(B.T), sigma, order)

    return K


def get_local_kernels_gaussian(
    A: ndarray, B: ndarray, na: ndarray, nb: ndarray, sigmas: List[float]
) -> ndarray:
    """Calculates the Gaussian kernel matrix K, for a local representation where :math:`K_{ij}`:

        :math:`K_{ij} = \\sum_{a \\in i} \\sum_{b \\in j} \\exp \\big( -\\frac{\\|A_a - B_b\\|_2^2}{2\\sigma^2} \\big)`

    Where :math:`A_{a}` and :math:`B_{b}` are representation vectors.

    Note that the input array is one big 2D array with all atoms concatenated along the same axis.
    Further more a series of kernels is produced (since calculating the distance matrix is expensive
    but getting the resulting kernels elements for several sigmas is not.)

    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of descriptors - shape (total atoms A, representation size).
    :type A: numpy array
    :param B: 2D array of descriptors - shape (total atoms B, representation size).
    :type B: numpy array
    :param na: 1D array containing numbers of atoms in each compound.
    :type na: numpy array
    :param nb: 1D array containing numbers of atoms in each compound.
    :type nb: numpy array
    :param sigma: The value of sigma in the kernel matrix.
    :type sigma: float

    :return: The Gaussian kernel matrix - shape (nsigmas, N, M)
    :rtype: numpy array
    """

    if np.sum(na) != A.shape[0]:
        raise ValueError("Error in A input")
    if np.sum(nb) != B.shape[0]:
        raise ValueError("Error in B input")

    if A.shape[1] != B.shape[1]:
        raise ValueError("Error in representation sizes")

    sigmas = np.asarray(sigmas)

    # Transpose for Fortran column-major format (3, n_atoms)
    return fget_local_kernels_gaussian(
        np.asfortranarray(A.T),
        np.asfortranarray(B.T),
        np.asfortranarray(na, dtype=np.int32),
        np.asfortranarray(nb, dtype=np.int32),
        np.asfortranarray(sigmas),
    )


def get_local_kernels_laplacian(
    A: ndarray, B: ndarray, na: ndarray, nb: ndarray, sigmas: List[float]
) -> ndarray:
    """Calculates the Local Laplacian kernel matrix K, for a local representation where :math:`K_{ij}`:

        :math:`K_{ij} = \\sum_{a \\in i} \\sum_{b \\in j} \\exp \\big( -\\frac{\\|A_a - B_b\\|_1}{\\sigma} \\big)`

    Where :math:`A_{a}` and :math:`B_{b}` are representation vectors.

    Note that the input array is one big 2D array with all atoms concatenated along the same axis.
    Further more a series of kernels is produced (since calculating the distance matrix is expensive
    but getting the resulting kernels elements for several sigmas is not.)

    K is calculated using an OpenMP parallel Fortran routine.

    :param A: 2D array of descriptors - shape (N, representation size).
    :type A: numpy array
    :param B: 2D array of descriptors - shape (M, representation size).
    :type B: numpy array
    :param na: 1D array containing numbers of atoms in each compound.
    :type na: numpy array
    :param nb: 1D array containing numbers of atoms in each compound.
    :type nb: numpy array
    :param sigmas: List of the sigmas.
    :type sigmas: list

    :return: The Laplacian kernel matrix - shape (nsigmas, N, M)
    :rtype: numpy array
    """

    if np.sum(na) != A.shape[0]:
        raise ValueError("Error in A input")
    if np.sum(nb) != B.shape[0]:
        raise ValueError("Error in B input")

    if A.shape[1] != B.shape[1]:
        raise ValueError("Error in representation sizes")

    sigmas = np.asarray(sigmas)

    # Transpose for Fortran column-major format (3, n_atoms)
    return fget_local_kernels_laplacian(
        np.asfortranarray(A.T),
        np.asfortranarray(B.T),
        np.asfortranarray(na, dtype=np.int32),
        np.asfortranarray(nb, dtype=np.int32),
        np.asfortranarray(sigmas),
    )


def kpca(K: ndarray, n: int = 2, centering: bool = True) -> ndarray:
    """Calculates `n` first principal components for the kernel :math:`K`.

    The PCA is calculated using an OpenMP parallel Fortran routine.

    A square, symmetric kernel matrix is required. Centering of the kernel matrix
    is enabled by default, although this isn't a strict requirement.

    :param K: 2D kernel matrix
    :type K: numpy array
    :param n: Number of kernel PCAs to return (default=2)
    :type n: integer
    :param centering: Whether to center the kernel matrix (default=True)
    :type centering: bool

    :return: array containing the principal components
    :rtype: numpy array
    """

    if K.shape[0] != K.shape[1]:
        raise ValueError("Square matrix required for Kernel PCA.")
    if not np.allclose(K, K.T, atol=1e-8):
        raise ValueError("Symmetric matrix required for Kernel PCA.")
    if not n <= K.shape[0]:
        raise ValueError("Requested more principal components than matrix size.")

    size = K.shape[0]
    pca = fkpca(K, size, centering)

    return pca[:n]

import numpy as np

from qmllib.utils.alchemy import get_alchemy

from .fchl_kernel_functions import get_kernel_parameters
from .ffchl_module import (
    fget_ef_atomic_local_gradient_kernels_fchl,
    fget_ef_gaussian_process_kernels_fchl,
)


def get_atomic_local_electric_field_gradient_kernels(
    A,
    B,
    verbose=False,
    df=0.01,
    ef_scaling=0.01,
    two_body_scaling=np.sqrt(8),
    three_body_scaling=1.6,
    two_body_width=0.2,
    three_body_width=np.pi,
    two_body_power=4.0,
    three_body_power=2.0,
    cut_start=1.0,
    cut_distance=5.0,
    fourier_order=1,
    alchemy="periodic-table",
    alchemy_period_width=1.6,
    alchemy_group_width=1.6,
    kernel="gaussian",
    kernel_args=None,
):
    """Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\\sigma^2} \\big)`

    Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
    K is calculated analytically using an OpenMP parallel Fortran routine.
    Note, that this kernel will ONLY work with FCHL representations as input.

    :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
    :type A: numpy array
    :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
    :type B: numpy array

    :param two_body_scaling: Weight for 2-body terms.
    :type two_body_scaling: float
    :param three_body_scaling: Weight for 3-body terms.
    :type three_body_scaling: float

    :param two_body_width: Gaussian width for 2-body terms
    :type two_body_width: float
    :param three_body_width: Gaussian width for 3-body terms.
    :type three_body_width: float

    :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
    :type two_body_power: float
    :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
    :type three_body_power: float

    :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
    :type cut_start: float
    :param cut_distance: Cut-off radius. (default=5 angstrom)
    :type cut_distance: float

    :param fourier_order: 3-body Fourier-expansion truncation order.
    :type fourier_order: integer
    :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
    :type alchemy: string

    :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
    :type alchemy_period_width: float
    :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
    :type alchemy_group_width: float

    :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
    :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    if not B.shape[1] == atoms_max:
        raise ValueError("Check FCHL representation sizes")
    if not B.shape[3] == neighbors_max:
        raise ValueError("Check FCHL representation sizes")

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1), dtype=np.int32)
    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a, i] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, n_kernels = get_kernel_parameters(kernel, kernel_args)

    na1 = np.sum(N1)

    return fget_ef_atomic_local_gradient_kernels_fchl(
        A,
        B,
        verbose,
        N1,
        N2,
        neighbors1,
        neighbors2,
        nm1,
        nm2,
        na1,
        n_kernels,
        three_body_width,
        two_body_width,
        cut_start,
        cut_distance,
        fourier_order,
        pd,
        two_body_scaling,
        three_body_scaling,
        doalchemy,
        two_body_power,
        three_body_power,
        ef_scaling,
        df,
        kernel_idx,
        kernel_parameters,
    )


def get_gaussian_process_electric_field_kernels(
    A,
    B,
    verbose=False,
    fields=None,
    df=0.01,
    ef_scaling=0.01,
    two_body_scaling=np.sqrt(8),
    three_body_scaling=1.6,
    two_body_width=0.2,
    three_body_width=np.pi,
    two_body_power=4.0,
    three_body_power=2.0,
    cut_start=1.0,
    cut_distance=5.0,
    fourier_order=1,
    alchemy="periodic-table",
    alchemy_period_width=1.6,
    alchemy_group_width=1.6,
    kernel="gaussian",
    kernel_args=None,
):
    """Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

        :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\\sigma^2} \\big)`

    Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
    K is calculated analytically using an OpenMP parallel Fortran routine.
    Note, that this kernel will ONLY work with FCHL representations as input.

    :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
    :type A: numpy array
    :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
    :type B: numpy array

    :param two_body_scaling: Weight for 2-body terms.
    :type two_body_scaling: float
    :param three_body_scaling: Weight for 3-body terms.
    :type three_body_scaling: float

    :param two_body_width: Gaussian width for 2-body terms
    :type two_body_width: float
    :param three_body_width: Gaussian width for 3-body terms.
    :type three_body_width: float

    :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
    :type two_body_power: float
    :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
    :type three_body_power: float

    :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
    :type cut_start: float
    :param cut_distance: Cut-off radius. (default=5 angstrom)
    :type cut_distance: float

    :param fourier_order: 3-body Fourier-expansion truncation order.
    :type fourier_order: integer
    :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
    :type alchemy: string

    :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
    :type alchemy_period_width: float
    :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
    :type alchemy_group_width: float

    :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
    :rtype: numpy array
    """

    atoms_max = A.shape[1]
    neighbors_max = A.shape[3]

    if not B.shape[1] == atoms_max:
        raise ValueError("Check FCHL representation sizes")
    if not B.shape[3] == neighbors_max:
        raise ValueError("Check FCHL representation sizes")

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    N1 = np.zeros((nm1), dtype=np.int32)
    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    for a, representation in enumerate(B):
        ni = N2[a]
        for i, x in enumerate(representation[:ni]):
            neighbors2[a, i] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, n_kernels = get_kernel_parameters(kernel, kernel_args)

    F1 = np.zeros((nm1, 3))
    F2 = np.zeros((nm2, 3))

    if fields is not None:

        F1 = np.array(fields[0])
        F2 = np.array(fields[1])

    return fget_ef_gaussian_process_kernels_fchl(
        A,
        B,
        verbose,
        F1,
        F2,
        N1,
        N2,
        neighbors1,
        neighbors2,
        nm1,
        nm2,
        n_kernels,
        three_body_width,
        two_body_width,
        cut_start,
        cut_distance,
        fourier_order,
        pd,
        two_body_scaling,
        three_body_scaling,
        doalchemy,
        two_body_power,
        three_body_power,
        ef_scaling,
        df,
        kernel_idx,
        kernel_parameters,
    )


def get_kernels_ef_field(
    A,
    B,
    fields=None,
    df=0.01,
    ef_scaling=0.01,
    two_body_scaling=np.sqrt(8),
    three_body_scaling=1.6,
    two_body_width=0.2,
    three_body_width=np.pi,
    two_body_power=4.0,
    three_body_power=2.0,
    cut_start=1.0,
    cut_distance=5.0,
    fourier_order=1,
    alchemy="periodic-table",
    alchemy_period_width=1.6,
    alchemy_group_width=1.6,
    kernel="gaussian",
    kernel_args=None,
):

    raise NotImplementedError("Undefined fget_kernels_fchl_ef_field")

    # atoms_max = A.shape[1]
    # neighbors_max = A.shape[3]

    # assert B.shape[1] == atoms_max, "Check FCHL representation sizes! code = 2"
    # assert B.shape[3] == neighbors_max, "Check FCHL representation sizes! code = 3"

    # nm1 = A.shape[0]
    # nm2 = B.shape[0]

    # N1 = np.zeros((nm1), dtype=np.int32)
    # N2 = np.zeros((nm2), dtype=np.int32)

    # for a in range(nm1):
    #     N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    # for a in range(nm2):
    #     N2[a] = len(np.where(B[a, :, 1, 0] > 0.0001)[0])

    # neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
    # neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

    # for a, representation in enumerate(A):
    #     ni = N1[a]
    #     for i, x in enumerate(representation[:ni]):
    #         neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    # for a, representation in enumerate(B):
    #     ni = N2[a]
    #     for i, x in enumerate(representation[:ni]):
    #         neighbors2[a, i] = len(np.where(x[0] < cut_distance)[0])

    # doalchemy, pd = get_alchemy(
    #     alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    # )

    # kernel_idx, kernel_parameters, n_kernels = get_kernel_parameters(kernel, kernel_args)

    # F2 = np.zeros((nm2, 3))

    # if fields is not None:

    #     F2 = np.array(fields)

    # na1 = np.sum(N1)

    # return fget_kernels_fchl_ef_field(
    #     A,
    #     B,
    #     F2,
    #     N1,
    #     N2,
    #     neighbors1,
    #     neighbors2,
    #     nm1,
    #     nm2,
    #     n_kernels,
    #     na1,
    #     three_body_width,
    #     two_body_width,
    #     cut_start,
    #     cut_distance,
    #     fourier_order,
    #     pd,
    #     two_body_scaling,
    #     three_body_scaling,
    #     doalchemy,
    #     two_body_power,
    #     three_body_power,
    #     ef_scaling,
    #     kernel_idx,
    #     kernel_parameters,
    # )


# Non-functional kernel for polarizabilites - unsure why it won't work

# def get_local_kernels_pol(A, B, df=1e-5, ef_scaling=1.0,
#         two_body_scaling=np.sqrt(8), three_body_scaling=1.6,
#         two_body_width=0.2, three_body_width=np.pi,
#         two_body_power=4.0, three_body_power=2.0,
#         cut_start=1.0, cut_distance=5.0,
#         fourier_order=1, alchemy="periodic-table",
#         alchemy_period_width=1.6, alchemy_group_width=1.6,
#         kernel="gaussian", kernel_args=None):
#     """ Calculates the Gaussian kernel matrix K, where :math:`K_{ij}`:

#             :math:`K_{ij} = \\exp \\big( -\\frac{\\|A_i - B_j\\|_2^2}{2\sigma^2} \\big)`

#         Where :math:`A_{i}` and :math:`B_{j}` are FCHL representation vectors.
#         K is calculated analytically using an OpenMP parallel Fortran routine.
#         Note, that this kernel will ONLY work with FCHL representations as input.

#         :param A: Array of FCHL representation - shape=(N, maxsize, 5, maxneighbors).
#         :type A: numpy array
#         :param B: Array of FCHL representation - shape=(M, maxsize, 5, maxneighbors).
#         :type B: numpy array

#         :param two_body_scaling: Weight for 2-body terms.
#         :type two_body_scaling: float
#         :param three_body_scaling: Weight for 3-body terms.
#         :type three_body_scaling: float

#         :param two_body_width: Gaussian width for 2-body terms
#         :type two_body_width: float
#         :param three_body_width: Gaussian width for 3-body terms.
#         :type three_body_width: float

#         :param two_body_power: Powerlaw for :math:`r^{-n}` 2-body terms.
#         :type two_body_power: float
#         :param three_body_power: Powerlaw for Axilrod-Teller-Muto 3-body term
#         :type three_body_power: float

#         :param cut_start: The fraction of the cut-off radius at which cut-off damping start.
#         :type cut_start: float
#         :param cut_distance: Cut-off radius. (default=5 angstrom)
#         :type cut_distance: float

#         :param fourier_order: 3-body Fourier-expansion truncation order.
#         :type fourier_order: integer
#         :param alchemy: Type of alchemical interpolation ``"periodic-table"`` or ``"off"`` are possible options. Disabling alchemical interpolation can yield dramatic speedups.
#         :type alchemy: string

#         :param alchemy_period_width: Gaussian width along periods (columns) in the periodic table.
#         :type alchemy_period_width: float
#         :param alchemy_group_width: Gaussian width along groups (rows) in the periodic table.
#         :type alchemy_group_width: float

#         :return: Array of FCHL kernel matrices matrix - shape=(n_sigmas, N, M),
#         :rtype: numpy array
#     """

#     atoms_max = A.shape[1]
#     neighbors_max = A.shape[3]

#     assert B.shape[1] == atoms_max, "Check FCHL representation sizes! code = 2"
#     assert B.shape[3] == neighbors_max, "Check FCHL representation sizes! code = 3"


#     nm1 = A.shape[0]
#     nm2 = B.shape[0]

#     N1 = np.zeros((nm1),dtype=np.int32)
#     N2 = np.zeros((nm2),dtype=np.int32)

#     for a in range(nm1):
#         N1[a] = len(np.where(A[a,:,1,0] > 0.0001)[0])

#     for a in range(nm2):
#         N2[a] = len(np.where(B[a,:,1,0] > 0.0001)[0])

#     neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)
#     neighbors2 = np.zeros((nm2, atoms_max), dtype=np.int32)

#     for a, representation in enumerate(A):
#         ni = N1[a]
#         for i, x in enumerate(representation[:ni]):
#             neighbors1[a,i] = len(np.where(x[0]< cut_distance)[0])

#     for a, representation in enumerate(B):
#         ni = N2[a]
#         for i, x in enumerate(representation[:ni]):
#             neighbors2[a,i] = len(np.where(x[0]< cut_distance)[0])

#     doalchemy, pd = get_alchemy(alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width)

#     kernel_idx, kernel_parameters, n_kernels = get_kernel_parameters(kernel, kernel_args)

#     na1 = np.sum(N1)

#     return fget_kernels_fchl_ef_2ndderiv(A, B, N1, N2, neighbors1, neighbors2, nm1, nm2, na1, n_kernels,
#                 three_body_width, two_body_width, cut_start, cut_distance, fourier_order, pd, two_body_scaling,
#                 three_body_scaling, doalchemy, two_body_power, three_body_power, ef_scaling, df, kernel_idx, kernel_parameters)

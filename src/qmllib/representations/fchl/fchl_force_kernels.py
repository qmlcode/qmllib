import numpy as np

from qmllib.utils.alchemy import get_alchemy

from .fchl_kernel_functions import get_kernel_parameters
from .ffchl_module import (
    fget_atomic_local_gradient_5point_kernels_fchl,
    fget_atomic_local_gradient_kernels_fchl,
    fget_force_alphas_fchl,
    fget_gaussian_process_kernels_fchl,
    fget_local_gradient_kernels_fchl,
    fget_local_hessian_kernels_fchl,
    fget_local_symmetric_hessian_kernels_fchl,
)


def get_gaussian_process_kernels(
    A,
    B,
    verbose=False,
    dx=0.005,
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

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    atoms_max = B.shape[4]
    neighbors_max = B.shape[6]

    if (
        not B.shape[1] == 3
        or not B.shape[2] == 2
        or not B.shape[5] == 5
        or not A.shape[2] == 5
        or not A.shape[1] == atoms_max
        or not B.shape[3] == atoms_max
        or not A.shape[3] == neighbors_max
    ):
        raise ValueError("Representations have wrong shape")

    N1 = np.zeros((nm1), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m, xyz, pm, i, :ni]):
                        neighbors2[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    naq2 = np.sum(N2) * 3

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    return fget_gaussian_process_kernels_fchl(
        A,
        B,
        verbose,
        N1,
        N2,
        neighbors1,
        neighbors2,
        nm1,
        nm2,
        naq2,
        nsigmas,
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
        dx,
        kernel_idx,
        kernel_parameters,
    )


def get_local_gradient_kernels(
    A,
    B,
    verbose=False,
    dx=0.005,
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

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    atoms_max = B.shape[4]

    neighbors_max = B.shape[6]

    if (
        not B.shape[1] == 3
        or not B.shape[2] == 2
        or not B.shape[5] == 5
        or not A.shape[2] == 5
        or not A.shape[1] == atoms_max
        or not B.shape[3] == atoms_max
        or not A.shape[3] == neighbors_max
    ):
        raise ValueError("Representations have wrong shape")

    N1 = np.zeros((nm1), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m, xyz, pm, i, :ni]):
                        neighbors2[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    naq2 = np.sum(N2) * 3

    return fget_local_gradient_kernels_fchl(
        A,
        B,
        verbose,
        N1,
        N2,
        neighbors1,
        neighbors2,
        nm1,
        nm2,
        naq2,
        nsigmas,
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
        dx,
        kernel_idx,
        kernel_parameters,
    )


def get_local_hessian_kernels(
    A,
    B,
    verbose=False,
    dx=0.005,
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

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    atoms_max = A.shape[4]

    neighbors_max = A.shape[6]

    if (
        not A.shape[1] == 3
        or not A.shape[2] == 2
        or not A.shape[5] == 5
        or not B.shape[1] == 3
        or not B.shape[2] == 2
        or not B.shape[5] == 5
        or not A.shape[3] == atoms_max
        or not B.shape[3] == atoms_max
        or not B.shape[6] == neighbors_max
    ):
        raise ValueError("Representations have wrong shape")

    N1 = np.zeros((nm1), dtype=np.int32)
    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    for a in range(nm2):
        N2[a] = len(np.where(B[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, 3, 2, atoms_max, atoms_max), dtype=np.int32)
    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm1):
        ni = N1[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(A[m, xyz, pm, i, :ni]):
                        neighbors1[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m, xyz, pm, i, :ni]):
                        neighbors2[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    naq1 = np.sum(N1) * 3
    naq2 = np.sum(N2) * 3

    # print naq1, naq2, nsigmas
    return fget_local_hessian_kernels_fchl(
        A,
        B,
        verbose,
        N1,
        N2,
        neighbors1,
        neighbors2,
        nm1,
        nm2,
        naq1,
        naq2,
        nsigmas,
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
        dx,
        kernel_idx,
        kernel_parameters,
    )


def get_local_symmetric_hessian_kernels(
    A,
    verbose=False,
    dx=0.005,
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

    nm1 = A.shape[0]

    atoms_max = A.shape[4]

    # Unused neighbors_max = A.shape[6]

    if (
        not A.shape[1] == 3
        or not A.shape[2] == 2
        or not A.shape[5] == 5
        or not A.shape[3] == atoms_max
    ):
        raise ValueError("Representations have wrong shape")

    N1 = np.zeros((nm1), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm1):
        ni = N1[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(A[m, xyz, pm, i, :ni]):
                        neighbors1[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    naq1 = np.sum(N1) * 3

    return fget_local_symmetric_hessian_kernels_fchl(
        A,
        verbose,
        N1,
        neighbors1,
        nm1,
        naq1,
        nsigmas,
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
        dx,
        kernel_idx,
        kernel_parameters,
    )


def get_force_alphas(
    A,
    B,
    F,
    verbose=False,
    energy=None,
    dx=0.005,
    regularization=1e-7,
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
    nm1 = A.shape[0]
    nm2 = B.shape[0]

    atoms_max = B.shape[4]

    neighbors_max = B.shape[6]

    if (
        not B.shape[1] == 3
        or not B.shape[2] == 2
        or not B.shape[5] == 5
        or not A.shape[2] == 5
        or not A.shape[1] == atoms_max
        or not B.shape[3] == atoms_max
        or not A.shape[3] == neighbors_max
    ):
        raise ValueError("Representations have wrong shape")

    N1 = np.zeros((nm1), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m, xyz, pm, i, :ni]):
                        neighbors2[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    na1 = np.sum(N1)
    # UNUSED naq2 = np.sum(N2) * 3

    E = np.zeros((nm1))
    if energy is not None:
        E = energy

    return fget_force_alphas_fchl(
        A,
        B,
        verbose,
        F,
        E,
        N1,
        N2,
        neighbors1,
        neighbors2,
        nm1,
        nm2,
        na1,
        nsigmas,
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
        dx,
        kernel_idx,
        kernel_parameters,
        regularization,
    )


def get_atomic_local_gradient_kernels(
    A,
    B,
    verbose=False,
    dx=0.005,
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

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    atoms_max = B.shape[4]

    neighbors_max = B.shape[6]

    if (
        not B.shape[1] == 3
        or not B.shape[2] == 2
        or not B.shape[5] == 5
        or not A.shape[2] == 5
        or not A.shape[1] == atoms_max
        or not B.shape[3] == atoms_max
        or not A.shape[3] == neighbors_max
    ):
        raise ValueError("Representations have wrong shape")

    N1 = np.zeros((nm1), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 2, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(2):
                for i in range(ni):
                    for a, x in enumerate(B[m, xyz, pm, i, :ni]):
                        neighbors2[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    na1 = np.sum(N1)
    naq2 = np.sum(N2) * 3

    return fget_atomic_local_gradient_kernels_fchl(
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
        naq2,
        nsigmas,
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
        dx,
        kernel_idx,
        kernel_parameters,
    )


def get_atomic_local_gradient_5point_kernels(
    A,
    B,
    verbose=False,
    dx=0.005,
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

    nm1 = A.shape[0]
    nm2 = B.shape[0]

    atoms_max = B.shape[4]

    neighbors_max = B.shape[6]

    if (
        not B.shape[1] == 3
        or not B.shape[2] == 5
        or not B.shape[5] == 5
        or not A.shape[2] == 5
        or not A.shape[1] == atoms_max
        or not B.shape[3] == atoms_max
        or not A.shape[3] == neighbors_max
    ):
        raise ValueError("Representations have wrong shape")

    N1 = np.zeros((nm1), dtype=np.int32)

    for a in range(nm1):
        N1[a] = len(np.where(A[a, :, 1, 0] > 0.0001)[0])

    neighbors1 = np.zeros((nm1, atoms_max), dtype=np.int32)

    for a, representation in enumerate(A):
        ni = N1[a]
        for i, x in enumerate(representation[:ni]):
            neighbors1[a, i] = len(np.where(x[0] < cut_distance)[0])

    N2 = np.zeros((nm2), dtype=np.int32)

    for a in range(nm2):
        N2[a] = len(np.where(B[a, 0, 0, 0, :, 1, 0] > 0.0001)[0])

    neighbors2 = np.zeros((nm2, 3, 5, atoms_max, atoms_max), dtype=np.int32)

    for m in range(nm2):
        ni = N2[m]
        for xyz in range(3):
            for pm in range(5):
                for i in range(ni):
                    for a, x in enumerate(B[m, xyz, pm, i, :ni]):
                        neighbors2[m, xyz, pm, i, a] = len(np.where(x[0] < cut_distance)[0])

    doalchemy, pd = get_alchemy(
        alchemy, emax=100, r_width=alchemy_group_width, c_width=alchemy_period_width
    )

    kernel_idx, kernel_parameters, nsigmas = get_kernel_parameters(kernel, kernel_args)

    na1 = np.sum(N1)
    naq2 = np.sum(N2) * 3

    return fget_atomic_local_gradient_5point_kernels_fchl(
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
        naq2,
        nsigmas,
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
        dx,
        kernel_idx,
        kernel_parameters,
    )

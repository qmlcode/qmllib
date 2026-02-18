import ast

import numpy as np
import pytest
from conftest import ASSETS
from scipy.stats import linregress

from qmllib.representations.fchl import (
    generate_fchl18,
    generate_fchl18_displaced,
    get_gaussian_process_kernels,
    get_local_gradient_kernels,
    get_local_hessian_kernels,
    get_local_kernels,
    get_local_symmetric_hessian_kernels,
    get_local_symmetric_kernels,
)
from qmllib.solvers import cho_solve

# Skip if pandas not installed
try:
    import pandas as pd
except ImportError:
    pytest.skip("pandas not installed", allow_module_level=True)


np.set_printoptions(linewidth=999, edgeitems=10, suppress=True)


TRAINING = 7
TEST = 5

ELEMENTS = [1, 6, 7, 8]

CUT_DISTANCE = 8.0

DF_TRAIN = pd.read_csv(ASSETS / "force_train.csv", delimiter=";").head(TRAINING)
DF_TEST = pd.read_csv(ASSETS / "force_test.csv", delimiter=";").head(TEST)

SIGMA = 2.5

LLAMBDA = 1e-6

np.random.seed(666)


def mae(a, b):
    return np.mean(np.abs(a.flatten() - b.flatten()))


def get_reps(df):
    x = []
    f = []
    e = []
    disp_x = []
    q = []

    CUT_DISTANCE = 1e6
    DX = 0.005
    max_atoms = 23
    for i in range(len(df)):
        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        # UNUSED atomtypes = df["atomtypes"][i]

        force = np.array(ast.literal_eval(df["forces"][i]))
        force *= -1

        energy = float(df["atomization_energy"][i])

        x1 = generate_fchl18(
            nuclear_charges,
            coordinates,
            max_size=max_atoms,
            cut_distance=CUT_DISTANCE,
        )

        dx1 = generate_fchl18_displaced(
            nuclear_charges,
            coordinates,
            max_size=max_atoms,
            cut_distance=CUT_DISTANCE,
            dx=DX,
        )

        x.append(x1)
        f.append(force)
        e.append(energy)

        disp_x.append(dx1)
        q.append(nuclear_charges)

    e = np.array(e)
    # e -= np.mean(e)# - 10 #

    # print(f)

    # f = np.array(f)
    # f *= -1
    x = np.array(x)

    return x, f, e, np.array(disp_x), q


def test_fchl_force():

    # Test that all kernel arguments work
    kernel_args = {
        "alchemy": "off",
        "kernel_args": {
            "sigma": [SIGMA],
        },
    }

    X, F, E, dX, Q = get_reps(DF_TRAIN)
    Xs, Fs, Es, dXs, Qs = get_reps(DF_TEST)

    F = np.concatenate(F)
    Fs = np.concatenate(Fs)

    Y = np.concatenate((E, F.flatten()))

    Kgp = get_gaussian_process_kernels(X, dX, dx=0.005, **kernel_args)[0]

    assert np.invert(np.all(np.isnan(Kgp))), "FCHL local kernel contains NaN"

    K_symmetric = get_local_symmetric_kernels(X, **kernel_args)[0]
    Kuu = Kgp[: len(X), : len(X)]
    assert np.allclose(K_symmetric, Kuu), "Error in FCHL local kernel and Gaussian process kernel"

    Kgrad = get_local_gradient_kernels(X, dX, **kernel_args)[0]
    Kgu = Kgp[len(X) :, : len(X)]
    assert np.allclose(Kgrad.T, Kgu), (
        "Error in FCHL local gradient kernel and Gaussian process kernel"
    )
    Kug = Kgp[: len(X), len(X) :]
    assert np.allclose(Kgrad, Kug), "Error in FCHL local gradient"

    Khess = get_local_symmetric_hessian_kernels(dX, dx=0.005, **kernel_args)[0]
    Kgg = Kgp[len(X) :, len(X) :]
    assert np.allclose(Khess, Kgg), "Error in FCHL local"

    Kgp[np.diag_indices_from(Kgp)] += LLAMBDA
    alpha = cho_solve(Kgp, Y)
    beta = alpha[:TRAINING]
    gamma = alpha[TRAINING:]

    Ks = get_local_hessian_kernels(dX, dXs, dx=0.005, **kernel_args)[0]
    Ks_energy = get_local_gradient_kernels(X, dXs, dx=0.005, **kernel_args)[0]

    Ks_energy2 = get_local_gradient_kernels(Xs, dX, dx=0.005, **kernel_args)[0]
    Ks_local = get_local_kernels(X, Xs, **kernel_args)[0]

    # Make predictions by manually combining kernel blocks
    # Test force predictions
    Fss = np.dot(np.transpose(Ks), gamma) + np.dot(Ks_energy.T, beta)
    # Training force predictions
    Kt = Kgp[TRAINING:, TRAINING:]
    Kt_energy = Kgp[:TRAINING, TRAINING:]
    Ft = np.dot(np.transpose(Kt), gamma) + np.dot(np.transpose(Kt_energy), beta)

    # Test energy predictions
    Ess = np.dot(Ks_energy2, gamma) + np.dot(Ks_local.T, beta)
    # Training energy predictions
    Kt_local = Kgp[:TRAINING, :TRAINING]
    Et = np.dot(Kt_energy, gamma) + np.dot(Kt_local.T, beta)

    # Print statistics (same format as test_fchl_acsf_gaussian_process)
    print(
        "==============================================================================================="
    )
    print(
        "====  GAUSSIAN PROCESS, FORCE + ENERGY (FCHL18 with force_train.csv)  ========================"
    )
    print(
        "==============================================================================================="
    )

    slope, intercept, r_value, p_value, std_err = linregress(E, Et)
    print(
        f"TRAINING ENERGY   MAE = {mae(Et, E):10.4f}  slope = {slope:10.4f}  intercept = {intercept:10.4f}  r^2 = {r_value:9.6f}"
    )

    slope, intercept, r_value, p_value, std_err = linregress(F.flatten(), Ft.flatten())
    print(
        f"TRAINING FORCE    MAE = {mae(Ft, F):10.4f}  slope = {slope:10.4f}  intercept = {intercept:10.4f}  r^2 = {r_value:9.6f}"
    )

    slope, intercept, r_value, p_value, std_err = linregress(Es.flatten(), Ess.flatten())
    print(
        f"TEST     ENERGY   MAE = {mae(Ess, Es):10.4f}  slope = {slope:10.4f}  intercept = {intercept:10.4f}  r^2 = {r_value:9.6f}"
    )

    slope, intercept, r_value, p_value, std_err = linregress(Fs.flatten(), Fss.flatten())
    print(
        f"TEST     FORCE    MAE = {mae(Fss, Fs):10.4f}  slope = {slope:10.4f}  intercept = {intercept:10.4f}  r^2 = {r_value:9.6f}"
    )

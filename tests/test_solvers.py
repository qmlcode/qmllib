from copy import deepcopy

import numpy as np
from conftest import ASSETS

from qmllib.solvers import bkf_invert, bkf_solve, cho_invert, cho_solve


def test_cho_solve():

    A_ref = np.loadtxt(ASSETS / "K_local_gaussian.txt")
    y_ref = np.loadtxt(ASSETS / "y_cho_solve.txt")

    A = deepcopy(A_ref)
    y = deepcopy(y_ref)
    x_qml = cho_solve(A, y)

    # Check arrays are unchanged
    assert np.allclose(y, y_ref)
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref)
    x_scipy = np.linalg.solve(A, y)

    # Check for correct solution
    assert np.allclose(x_qml, x_scipy)


def test_cho_invert():

    A_ref = np.loadtxt(ASSETS / "K_local_gaussian.txt")

    A = deepcopy(A_ref)
    Ai_qml = cho_invert(A)

    # Check A is unchanged
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref)
    one = np.eye(A.shape[0])

    # Check that it is a true inverse
    assert np.allclose(np.matmul(A, Ai_qml), one, atol=1e-7)


def test_bkf_invert():

    A_ref = np.loadtxt(ASSETS / "K_local_gaussian.txt")

    A = deepcopy(A_ref)
    Ai_qml = bkf_invert(A)

    # Check A is unchanged
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref)
    one = np.eye(A.shape[0])

    np.set_printoptions(linewidth=20000)
    assert np.allclose(np.matmul(A, Ai_qml), one, atol=1e-7)


def test_bkf_solve():

    A_ref = np.loadtxt(ASSETS / "K_local_gaussian.txt")
    y_ref = np.loadtxt(ASSETS / "y_cho_solve.txt")

    A = deepcopy(A_ref)
    y = deepcopy(y_ref)
    x_qml = bkf_solve(A, y)

    # Check arrays are unchanged
    assert np.allclose(y, y_ref)
    assert np.allclose(A, A_ref)

    A = deepcopy(A_ref)
    y = deepcopy(y_ref)
    x_scipy = np.linalg.solve(A, y)

    # Check for correct solution
    assert np.allclose(x_qml, x_scipy)


if __name__ == "__main__":
    test_cho_solve()
    test_cho_invert()
    test_bkf_invert()
    test_bkf_solve()

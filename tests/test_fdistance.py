import numpy as np
from qmllib._fdistance import (
    fmanhattan_distance,
    fl2_distance,
    fp_distance_double,
    fp_distance_integer,
)


def test_manhattan_distance():
    """Test Manhattan (L1) distance function"""
    np.random.seed(42)
    nv = 10
    na = 5
    nb = 7

    A = np.random.rand(nv, na).astype(np.float64, order="F")
    B = np.random.rand(nv, nb).astype(np.float64, order="F")

    D = fmanhattan_distance(A, B)

    assert D.shape == (na, nb), f"Wrong shape: {D.shape}"

    # Verify correctness
    manual = np.zeros((na, nb))
    for i in range(na):
        for j in range(nb):
            manual[i, j] = np.sum(np.abs(A[:, i] - B[:, j]))

    assert np.allclose(D, manual), "Manhattan distance incorrect!"


def test_l2_distance():
    """Test L2 (Euclidean) distance function"""
    np.random.seed(123)
    nv = 8
    na = 4
    nb = 6

    A = np.random.rand(nv, na).astype(np.float64, order="F")
    B = np.random.rand(nv, nb).astype(np.float64, order="F")

    D = fl2_distance(A, B)

    assert D.shape == (na, nb), f"Wrong shape: {D.shape}"

    # Verify correctness
    manual = np.zeros((na, nb))
    for i in range(na):
        for j in range(nb):
            manual[i, j] = np.sqrt(np.sum((A[:, i] - B[:, j]) ** 2))

    assert np.allclose(D, manual), "L2 distance incorrect!"


def test_lp_distance_double():
    """Test Lp distance with double precision p"""
    np.random.seed(456)
    nv = 6
    na = 3
    nb = 4
    p = 2.5

    A = np.random.rand(nv, na).astype(np.float64, order="F")
    B = np.random.rand(nv, nb).astype(np.float64, order="F")

    D = fp_distance_double(A, B, p)

    assert D.shape == (na, nb), f"Wrong shape: {D.shape}"

    # Verify correctness
    manual = np.zeros((na, nb))
    for i in range(na):
        for j in range(nb):
            manual[i, j] = (np.sum(np.abs(A[:, i] - B[:, j]) ** p)) ** (1.0 / p)

    assert np.allclose(D, manual), "Lp distance (double) incorrect!"


def test_lp_distance_integer():
    """Test Lp distance with integer p"""
    np.random.seed(789)
    nv = 7
    na = 5
    nb = 5
    p = 3

    A = np.random.rand(nv, na).astype(np.float64, order="F")
    B = np.random.rand(nv, nb).astype(np.float64, order="F")

    D = fp_distance_integer(A, B, p)

    assert D.shape == (na, nb), f"Wrong shape: {D.shape}"

    # Verify correctness
    manual = np.zeros((na, nb))
    for i in range(na):
        for j in range(nb):
            manual[i, j] = (np.sum(np.abs(A[:, i] - B[:, j]) ** p)) ** (1.0 / p)

    assert np.allclose(D, manual), "Lp distance (integer) incorrect!"


def test_distance_symmetry():
    """Test that distance(A, B) has correct symmetry properties"""
    np.random.seed(999)
    nv = 5
    n = 6

    A = np.random.rand(nv, n).astype(np.float64, order="F")

    # Distance from A to itself should have symmetric matrix
    D_manhattan = fmanhattan_distance(A, A)
    assert np.allclose(D_manhattan, D_manhattan.T), "Manhattan distance not symmetric!"

    D_l2 = fl2_distance(A, A)
    assert np.allclose(D_l2, D_l2.T), "L2 distance not symmetric!"

    # Diagonal should be zero (distance from point to itself)
    assert np.allclose(np.diag(D_manhattan), 0), "Manhattan distance diagonal not zero!"
    assert np.allclose(np.diag(D_l2), 0), "L2 distance diagonal not zero!"

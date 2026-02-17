import numpy as np
import pytest
from conftest import ASSETS, get_energies
from scipy.stats import wasserstein_distance

# Skip if sklearn not installed
try:
    from sklearn.decomposition import KernelPCA
except ImportError:
    pytest.skip("sklearn not installed", allow_module_level=True)

from qmllib._fkernels import fkpca, fwasserstein_kernel
from qmllib.representations import generate_bob
from qmllib.utils.xyz_format import read_xyz


def array_nan_close(a, b):
    # Compares arrays, ignoring nans
    m = np.isfinite(a) & np.isfinite(b)
    return np.allclose(a[m], b[m], atol=1e-8, rtol=0.0)


def test_kpca():
    """Test kernel PCA function"""
    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filename
    data = get_energies(ASSETS / "hof_qm7.txt")

    keys = sorted(data.keys())

    np.random.seed(666)
    np.random.shuffle(keys)

    n_mols = 100

    representations = []

    for xyz_file in keys[:n_mols]:
        filename = ASSETS / "qm7" / xyz_file
        coordinates, atoms = read_xyz(filename)

        atomtypes = np.unique(atoms)
        representation = generate_bob(atoms, coordinates, atomtypes)
        representations.append(representation)

    X = np.array([representation for representation in representations])

    # Calculate laplacian kernel manually (since fkernels not converted yet)
    sigma = 2e5
    na = X.shape[0]
    K = np.empty((na, na), order="F")

    for i in range(na):
        for j in range(na):
            K[i, j] = np.exp(-np.sum(np.abs(X[i] - X[j])) / sigma)

    K = np.asfortranarray(K)

    # Calculate PCA using our pybind11 function
    n_components = 10
    pcas_qml = fkpca(K, K.shape[0], centering=1)[:n_components]

    # Calculate with sklearn
    pcas_sklearn = KernelPCA(
        10, eigen_solver="dense", kernel="precomputed"
    ).fit_transform(K)

    assert array_nan_close(np.abs(pcas_sklearn.T), np.abs(pcas_qml)), (
        "Error in Kernel PCA decomposition."
    )


def test_wasserstein_kernel():
    """Test Wasserstein kernel function"""
    np.random.seed(666)

    n_train = 5
    n_test = 3

    # List of dummy representations (rep_size x n)
    rep_size = 3
    X = np.array(
        np.random.randint(0, 10, size=(rep_size, n_train)), dtype=np.float64, order="F"
    )
    Xs = np.array(
        np.random.randint(0, 10, size=(rep_size, n_test)), dtype=np.float64, order="F"
    )

    sigma = 100.0

    Ktest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            Ktest[i, j] = np.exp(
                wasserstein_distance(X[:, i], Xs[:, j]) / (-1.0 * sigma)
            )

    K = fwasserstein_kernel(X, n_train, Xs, n_test, sigma, 1, 1)

    # Compare two implementations:
    assert np.allclose(K, Ktest), "Error in Wasserstein kernel"

    Ksymm = fwasserstein_kernel(X, n_train, X, n_train, sigma, 1, 1)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm.T), "Error in Wasserstein kernel symmetry"


def test_kpca_no_centering():
    """Test KPCA without centering"""
    np.random.seed(42)
    n = 20
    # Create a positive definite matrix using X.T @ X
    X = np.random.rand(n, n + 10)
    K = X @ X.T  # This is guaranteed to be positive semidefinite
    K = np.asfortranarray(K)

    # Test without centering
    pca_result = fkpca(K, n, centering=0)

    assert pca_result.shape == (n, n), f"Wrong shape: {pca_result.shape}"

    # Check that result is finite
    assert np.all(np.isfinite(pca_result)), "KPCA result contains NaN or Inf"


def test_wasserstein_different_p_q():
    """Test Wasserstein kernel with different p and q parameters"""
    np.random.seed(123)

    rep_size = 4
    na = 6
    nb = 4

    A = np.random.rand(rep_size, na).astype(np.float64, order="F")
    B = np.random.rand(rep_size, nb).astype(np.float64, order="F")

    sigma = 50.0

    # Test with p=2, q=1
    K1 = fwasserstein_kernel(A, na, B, nb, sigma, 2, 1)
    assert K1.shape == (na, nb), f"Wrong shape: {K1.shape}"
    assert np.all(np.isfinite(K1)), "Kernel contains NaN or Inf"
    assert np.all(K1 > 0) and np.all(K1 <= 1), "Kernel values outside expected range"

    # Test with p=1, q=2
    K2 = fwasserstein_kernel(A, na, B, nb, sigma, 1, 2)
    assert K2.shape == (na, nb), f"Wrong shape: {K2.shape}"
    assert np.all(np.isfinite(K2)), "Kernel contains NaN or Inf"

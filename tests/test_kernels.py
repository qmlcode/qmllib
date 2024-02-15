import numpy as np
from conftest import ASSETS, get_energies
from scipy.stats import wasserstein_distance
from sklearn.decomposition import KernelPCA

from qmllib.kernels import (
    gaussian_kernel,
    gaussian_kernel_symmetric,
    kpca,
    laplacian_kernel,
    laplacian_kernel_symmetric,
    linear_kernel,
    matern_kernel,
    sargan_kernel,
    wasserstein_kernel,
)
from qmllib.representations import generate_bob
from qmllib.utils.xyz_format import read_xyz


def test_laplacian_kernel():

    np.random.seed(666)

    n_train = 25
    n_test = 20

    # List of dummy representations
    X = np.random.rand(n_train, 1000)
    Xs = np.random.rand(n_test, 1000)

    sigma = 100.0

    Ktest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            Ktest[i, j] = np.exp(np.sum(np.abs(X[i] - Xs[j])) / (-1.0 * sigma))

    K = laplacian_kernel(X, Xs, sigma)

    # Compare two implementations:
    assert np.allclose(K, Ktest), "Error in Laplacian kernel"

    Ksymm = laplacian_kernel(X, X, sigma)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm.T), "Error in Laplacian kernel"

    Ksymm2 = laplacian_kernel_symmetric(X, sigma)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm2), "Error in Laplacian kernel"


def test_gaussian_kernel():

    np.random.seed(666)

    n_train = 25
    n_test = 20

    # List of dummy representations
    X = np.random.rand(n_train, 1000)
    Xs = np.random.rand(n_test, 1000)

    sigma = 100.0

    Ktest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            Ktest[i, j] = np.exp(np.sum(np.square(X[i] - Xs[j])) / (-2.0 * sigma**2))

    K = gaussian_kernel(X, Xs, sigma)

    # Compare two implementations:
    assert np.allclose(K, Ktest), "Error in Gaussian kernel"

    Ksymm = gaussian_kernel(X, X, sigma)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm.T), "Error in Gaussian kernel"

    Ksymm2 = gaussian_kernel_symmetric(X, sigma)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm2), "Error in Gaussian kernel"


def test_linear_kernel():

    np.random.seed(666)

    n_train = 25
    n_test = 20

    # List of dummy representations
    X = np.random.rand(n_train, 1000)
    Xs = np.random.rand(n_test, 1000)

    # UNUSED sigma = 100.0

    Ktest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            Ktest[i, j] = np.dot(X[i], Xs[j])

    K = linear_kernel(X, Xs)

    # Compare two implementations:
    assert np.allclose(K, Ktest), "Error in linear kernel"

    Ksymm = linear_kernel(X, X)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm.T), "Error in linear kernel"


def test_matern_kernel():

    np.random.seed(666)

    for metric in ("l1", "l2"):
        for order in (0, 1, 2):
            matern(metric, order)


def matern(metric, order):

    n_train = 25
    n_test = 20

    # List of dummy representations
    X = np.random.rand(n_train, 1000)
    Xs = np.random.rand(n_test, 1000)

    sigma = 100.0

    Ktest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):

            if metric == "l1":
                d = np.sum(abs(X[i] - Xs[j]))
            else:
                d = np.sqrt(np.sum((X[i] - Xs[j]) ** 2))

            if order == 0:
                Ktest[i, j] = np.exp(-d / sigma)
            elif order == 1:
                Ktest[i, j] = np.exp(-np.sqrt(3) * d / sigma) * (1 + np.sqrt(3) * d / sigma)
            else:
                Ktest[i, j] = np.exp(-np.sqrt(5) * d / sigma) * (
                    1 + np.sqrt(5) * d / sigma + 5.0 / 3 * d**2 / sigma**2
                )

    K = matern_kernel(X, Xs, sigma, metric=metric, order=order)

    # Compare two implementations:
    assert np.allclose(K, Ktest), "Error in Matern kernel"

    Ksymm = matern_kernel(X, X, sigma, metric=metric, order=order)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm.T), "Error in Matern kernel"


def test_sargan_kernel():

    np.random.seed(666)

    for ngamma in (0, 1, 2):
        sargan(ngamma)


def sargan(ngamma):

    n_train = 25
    n_test = 20

    gammas = np.random.random(ngamma)

    # List of dummy representations
    X = np.random.rand(n_train, 1000)
    Xs = np.random.rand(n_test, 1000)

    sigma = 100.0

    Ktest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            d = np.sum(abs(X[i] - Xs[j]))

            factor = 1
            for k, gamma in enumerate(gammas):
                factor += gamma / sigma ** (k + 1) * d ** (k + 1)
            Ktest[i, j] = np.exp(-d / sigma) * factor

    K = sargan_kernel(X, Xs, sigma, gammas)

    # Compare two implementations:
    assert np.allclose(K, Ktest), "Error in Sargan kernel"

    Ksymm = sargan_kernel(X, X, sigma, gammas)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm.T), "Error in Sargan kernel"


def array_nan_close(a, b):
    # Compares arrays, ignoring nans

    m = np.isfinite(a) & np.isfinite(b)
    return np.allclose(a[m], b[m], atol=1e-8, rtol=0.0)


def test_kpca():

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenam
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
    K = laplacian_kernel(X, X, 2e5)

    # calculate pca
    pcas_qml = kpca(K, n=10)

    # Calculate with sklearn
    pcas_sklearn = KernelPCA(10, eigen_solver="dense", kernel="precomputed").fit_transform(K)

    assert array_nan_close(
        np.abs(pcas_sklearn.T), np.abs(pcas_qml)
    ), "Error in Kernel PCA decomposition."


def test_wasserstein_kernel():

    np.random.seed(666)

    n_train = 5
    n_test = 3

    # List of dummy representations
    X = np.array(np.random.randint(0, 10, size=(n_train, 3)), dtype=np.float64)
    Xs = np.array(np.random.randint(0, 10, size=(n_test, 3)), dtype=np.float64)

    sigma = 100.0

    Ktest = np.zeros((n_train, n_test))

    for i in range(n_train):
        for j in range(n_test):
            Ktest[i, j] = np.exp(wasserstein_distance(X[i], Xs[j]) / (-1.0 * sigma))

    K = wasserstein_kernel(X, Xs, sigma)

    # Compare two implementations:
    assert np.allclose(K, Ktest), "Error in Wasserstein kernel"

    Ksymm = wasserstein_kernel(X, X, sigma)

    # Check for symmetry:
    assert np.allclose(Ksymm, Ksymm.T), "Error in Wasserstein kernel"

import numpy as np

from qmllib.kernels.kernels import (
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


def test_linear_kernel():
    """Linear kernel should be equivalent to matrix multiplication (dot product)."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    K = linear_kernel(A, B)
    expected = A @ B.T

    assert K.shape == (2, 2)
    assert np.allclose(K, expected)


def test_linear_kernel_square():
    """Linear kernel with same input should produce symmetric matrix."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    K = linear_kernel(A, A)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)  # Should be symmetric


def test_gaussian_kernel_identity():
    """Gaussian kernel of identical points should be 1."""
    A = np.array([[1.0, 2.0, 3.0]])

    K = gaussian_kernel(A, A, sigma=1.0)

    assert K.shape == (1, 1)
    assert np.isclose(K[0, 0], 1.0), "Same point should have kernel value 1"


def test_gaussian_kernel_shape():
    """Gaussian kernel should produce correct output shape."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    K = gaussian_kernel(A, B, sigma=1.0)

    assert K.shape == (2, 3)
    assert np.all(K >= 0) and np.all(K <= 1), "Gaussian kernel values should be in [0,1]"


def test_gaussian_kernel_symmetric():
    """Symmetric Gaussian kernel should produce symmetric matrix."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    K = gaussian_kernel_symmetric(A, sigma=1.0)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T), "Symmetric kernel should produce symmetric matrix"
    assert np.allclose(np.diag(K), 1.0), "Diagonal should be all 1s"


def test_laplacian_kernel_identity():
    """Laplacian kernel of identical points should be 1."""
    A = np.array([[1.0, 2.0, 3.0]])

    K = laplacian_kernel(A, A, sigma=1.0)

    assert K.shape == (1, 1)
    assert np.isclose(K[0, 0], 1.0), "Same point should have kernel value 1"


def test_laplacian_kernel_shape():
    """Laplacian kernel should produce correct output shape."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    K = laplacian_kernel(A, B, sigma=1.0)

    assert K.shape == (2, 3)
    assert np.all(K >= 0) and np.all(K <= 1), "Laplacian kernel values should be in [0,1]"


def test_laplacian_kernel_symmetric():
    """Symmetric Laplacian kernel should produce symmetric matrix."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    K = laplacian_kernel_symmetric(A, sigma=1.0)

    assert K.shape == (3, 3)
    assert np.allclose(K, K.T), "Symmetric kernel should produce symmetric matrix"
    assert np.allclose(np.diag(K), 1.0), "Diagonal should be all 1s"


def test_matern_kernel_basic():
    """Matérn kernel should produce valid kernel matrix."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    # Test with order=0 (equivalent to Laplacian)
    K = matern_kernel(A, B, sigma=1.0, order=0, metric="l1")

    assert K.shape == (2, 2)
    assert np.all(K >= 0) and np.all(K <= 1), "Matérn kernel values should be in [0,1]"


def test_matern_kernel_identity():
    """Matérn kernel of identical points should be 1."""
    A = np.array([[1.0, 2.0]])

    K = matern_kernel(A, A, sigma=1.0, order=1, metric="l2")

    assert K.shape == (1, 1)
    assert np.isclose(K[0, 0], 1.0), "Same point should have kernel value 1"


def test_sargan_kernel_basic():
    """Sargan kernel should produce valid kernel matrix."""
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])

    gammas = np.array([0.5, 1.0])
    K = sargan_kernel(A, B, sigma=1.0, gammas=gammas)

    assert K.shape == (2, 2)
    assert np.all(np.isfinite(K)), "Sargan kernel should not contain NaN/Inf"


def test_sargan_kernel_identity():
    """Sargan kernel of identical points should be 1."""
    A = np.array([[1.0, 2.0]])

    gammas = np.array([1.0])
    K = sargan_kernel(A, A, sigma=1.0, gammas=gammas)

    assert K.shape == (1, 1)
    assert np.isclose(K[0, 0], 1.0), "Same point should have kernel value 1"


def test_wasserstein_kernel_basic():
    """Wasserstein kernel should produce valid kernel matrix."""
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    K = wasserstein_kernel(A, B, sigma=1.0, p=1, q=1)

    assert K.shape == (2, 2)
    assert np.all(K >= 0) and np.all(K <= 1), "Wasserstein kernel values should be in [0,1]"


def test_wasserstein_kernel_identity():
    """Wasserstein kernel of identical points should be 1."""
    A = np.array([[1.0, 2.0, 3.0]])

    K = wasserstein_kernel(A, A, sigma=1.0, p=1, q=1)

    assert K.shape == (1, 1)
    assert np.isclose(K[0, 0], 1.0, atol=1e-10), "Same point should have kernel value 1"


def test_kpca_basic():
    """KPCA should reduce dimensionality of kernel matrix."""
    # Create a simple kernel matrix
    K = np.array(
        [[1.0, 0.8, 0.6, 0.4], [0.8, 1.0, 0.7, 0.5], [0.6, 0.7, 1.0, 0.6], [0.4, 0.5, 0.6, 1.0]]
    )

    # Reduce to 2 dimensions
    X_reduced = kpca(K, n=2, centering=True)

    assert X_reduced.shape == (2, 4), "KPCA should return (n_components, n_samples)"
    assert np.all(np.isfinite(X_reduced)), "KPCA output should not contain NaN/Inf"


def test_kpca_no_centering():
    """KPCA should work without centering."""
    K = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])

    X_reduced = kpca(K, n=2, centering=False)

    assert X_reduced.shape == (2, 3), "KPCA should return (n_components, n_samples)"
    assert np.all(np.isfinite(X_reduced))


if __name__ == "__main__":
    # Run all tests
    test_linear_kernel()
    test_linear_kernel_square()
    test_gaussian_kernel_identity()
    test_gaussian_kernel_shape()
    test_gaussian_kernel_symmetric()
    test_laplacian_kernel_identity()
    test_laplacian_kernel_shape()
    test_laplacian_kernel_symmetric()
    test_matern_kernel_basic()
    test_matern_kernel_identity()
    test_sargan_kernel_basic()
    test_sargan_kernel_identity()
    test_wasserstein_kernel_basic()
    test_wasserstein_kernel_identity()
    test_kpca_basic()
    test_kpca_no_centering()
    print("All kernel tests passed!")

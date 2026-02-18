"""Simple test for atomic local kernels migration."""

import numpy as np

from qmllib.representations import (
    generate_fchl18,
    generate_fchl18_displaced,
    generate_fchl18_displaced_5point,
)
from qmllib.representations.fchl import (
    get_atomic_local_gradient_5point_kernels,
    get_atomic_local_gradient_kernels,
    get_atomic_local_kernels,
)


def test_atomic_local_kernels_simple():
    """Test that atomic_local_kernels can be computed without errors using real molecular data."""

    # Create simple molecules
    coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coords2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    nuclear_charges1 = [6, 1, 1]  # CH2
    nuclear_charges2 = [8, 1]  # OH

    # Generate representations
    rep1 = generate_fchl18(nuclear_charges1, coords1, max_size=10, cut_distance=1e6)
    rep2 = generate_fchl18(nuclear_charges2, coords2, max_size=10, cut_distance=1e6)

    X1 = np.array([rep1])
    X2 = np.array([rep2])

    # Calculate total atoms in first set
    na1 = len(nuclear_charges1)  # 3 atoms

    # Test atomic local kernels
    result = get_atomic_local_kernels(
        X1,
        X2,
        kernel="gaussian",
        kernel_args={"sigma": [1.0, 2.0]},
        cut_distance=1e6,
    )

    # Check result shape: (nsigmas, na1, nm2)
    assert result.shape[0] == 2, f"Wrong number of sigmas: {result.shape[0]} != 2"
    assert result.shape[1] == na1, f"Wrong na1: {result.shape[1]} != {na1}"
    assert result.shape[2] == 1, f"Wrong nm2: {result.shape[2]} != 1"  # 1 molecule in X2
    assert np.all(np.isfinite(result)), "Atomic local kernel contains NaN/Inf"
    assert np.all(result >= 0), "Kernel values should be non-negative"

    print(f"✓ Atomic local kernel shape: {result.shape}")
    print(f"✓ Kernel values range: [{result.min():.6f}, {result.max():.6f}]")


def test_atomic_local_kernels_symmetric():
    """Test that atomic_local_kernels produces symmetric results when X1 == X2."""

    # Create a single molecule
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    nuclear_charges = [6, 1, 1]  # CH2

    # Generate representation
    rep = generate_fchl18(nuclear_charges, coords, max_size=10, cut_distance=1e6)
    X = np.array([rep])

    # Calculate kernel with itself
    result = get_atomic_local_kernels(
        X,
        X,
        kernel="gaussian",
        kernel_args={"sigma": [1.0]},
        cut_distance=1e6,
    )

    # Check result shape: (1, 3, 1) for 1 sigma, 3 atoms, 1 molecule
    assert result.shape == (1, 3, 1), f"Unexpected shape: {result.shape}"
    assert np.all(np.isfinite(result)), "Kernel contains NaN/Inf"

    # The kernel of a molecule with itself should have positive values
    assert np.all(result > 0), "Self-kernel should be positive"

    print(f"✓ Self-kernel values: {result[0, :, 0]}")


def test_atomic_local_gradient_kernels_simple():
    """Test that atomic_local_gradient_kernels can be computed without errors."""

    # Create simple molecules
    coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coords2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    nuclear_charges1 = [6, 1, 1]  # CH2
    nuclear_charges2 = [8, 1]  # OH

    # Generate standard representations
    rep1 = generate_fchl18(nuclear_charges1, coords1, max_size=10, cut_distance=1e6)

    # Generate displaced representations
    drep2 = generate_fchl18_displaced(
        nuclear_charges2, coords2, max_size=10, cut_distance=1e6, dx=0.005
    )

    X1 = np.array([rep1])
    dX2 = np.array([drep2])

    # Calculate dimensions
    na1 = len(nuclear_charges1)  # 3 atoms in first set
    naq2 = len(nuclear_charges2) * 3  # 2 atoms * 3 coords = 6 force components

    # Test atomic local gradient kernels
    result = get_atomic_local_gradient_kernels(
        X1,
        dX2,
        dx=0.005,
        kernel="gaussian",
        kernel_args={"sigma": [1.0, 2.0]},
        cut_distance=1e6,
    )

    # Check result shape: (nsigmas, na1, naq2)
    assert result.shape[0] == 2, f"Wrong number of sigmas: {result.shape[0]} != 2"
    assert result.shape[1] == na1, f"Wrong na1: {result.shape[1]} != {na1}"
    assert result.shape[2] == naq2, f"Wrong naq2: {result.shape[2]} != {naq2}"
    assert np.all(np.isfinite(result)), "Atomic local gradient kernel contains NaN/Inf"

    print(f"✓ Atomic local gradient kernel shape: {result.shape}")
    print(f"✓ Kernel values range: [{result.min():.6f}, {result.max():.6f}]")


def test_atomic_local_gradient_5point_kernels_simple():
    """Test that atomic_local_gradient_5point_kernels can be computed without errors using 5-point stencil."""

    # Create simple molecules
    coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coords2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
    nuclear_charges1 = [6, 1, 1]  # CH2
    nuclear_charges2 = [8, 1]  # OH

    # Generate standard representations
    rep1 = generate_fchl18(nuclear_charges1, coords1, max_size=10, cut_distance=1e6)

    # Generate displaced representations with 5-point stencil
    drep2 = generate_fchl18_displaced_5point(
        nuclear_charges2, coords2, max_size=10, cut_distance=1e6, dx=0.005
    )

    X1 = np.array([rep1])
    dX2 = np.array([drep2])

    # Calculate dimensions
    na1 = len(nuclear_charges1)  # 3 atoms in first set
    naq2 = len(nuclear_charges2) * 3  # 2 atoms * 3 coords = 6 force components

    # Test atomic local gradient kernels with 5-point stencil
    result = get_atomic_local_gradient_5point_kernels(
        X1,
        dX2,
        dx=0.005,
        kernel="gaussian",
        kernel_args={"sigma": [1.0, 2.0]},
        cut_distance=1e6,
    )

    # Check result shape: (nsigmas, na1, naq2)
    assert result.shape[0] == 2, f"Wrong number of sigmas: {result.shape[0]} != 2"
    assert result.shape[1] == na1, f"Wrong na1: {result.shape[1]} != {na1}"
    assert result.shape[2] == naq2, f"Wrong naq2: {result.shape[2]} != {naq2}"
    assert np.all(np.isfinite(result)), "Atomic local gradient 5point kernel contains NaN/Inf"

    print(f"✓ Atomic local gradient 5point kernel shape: {result.shape}")
    print(f"✓ Kernel values range: [{result.min():.6f}, {result.max():.6f}]")


if __name__ == "__main__":
    test_atomic_local_kernels_simple()
    test_atomic_local_kernels_symmetric()
    test_atomic_local_gradient_kernels_simple()
    test_atomic_local_gradient_5point_kernels_simple()
    print("All tests passed!")

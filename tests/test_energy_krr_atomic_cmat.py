import numpy as np
from conftest import ASSETS, get_energies, shuffle_arrays

from qmllib.kernels import get_local_kernels_gaussian, get_local_kernels_laplacian
from qmllib.representations import generate_atomic_coulomb_matrix
from qmllib.solvers import cho_solve
from qmllib.utils.xyz_format import read_xyz


def test_krr_gaussian_local_cmat():

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(ASSETS / "hof_qm7.txt")

    n_points = 1000

    all_representations = []
    all_properties = []
    all_atoms = []

    filenames = sorted(data.keys())[:n_points]

    for filename in filenames:
        coord, atoms = read_xyz((ASSETS / "qm7" / filename).with_suffix(".xyz"))

        representation = generate_atomic_coulomb_matrix(atoms, coord, size=23, sorting="row-norm")

        all_representations.append(representation)
        all_properties.append(data[filename])
        all_atoms.append(atoms)

    all_properties = np.array(all_properties)

    shuffle_arrays(all_atoms, all_properties, all_representations, seed=666)

    # Make training and test sets
    n_test = 100
    n_train = 200
    indices = list(range(n_points))
    train_indices = indices[:n_train]
    test_indices = indices[-n_test:]

    train_representations = np.concatenate([all_representations[i] for i in train_indices])
    test_representations = np.concatenate([all_representations[i] for i in test_indices])

    test_atoms = [all_atoms[x] for x in test_indices]
    test_properties = all_properties[test_indices]

    train_atoms = [all_atoms[x] for x in train_indices]
    train_properties = all_properties[train_indices]

    train_sizes = np.array([len(atoms) for atoms in train_atoms])
    test_sizes = np.array([len(atoms) for atoms in test_atoms])

    # Set hyper-parameters
    sigma = 724.0
    llambda = 10 ** (-6.5)

    K = get_local_kernels_gaussian(
        train_representations, train_representations, train_sizes, train_sizes, [sigma]
    )[0]
    assert np.allclose(K, K.T), "Error in local Gaussian kernel symmetry"

    K_test = np.loadtxt(ASSETS / "K_local_gaussian.txt")
    assert np.allclose(K, K_test), "Error in local Gaussian kernel (vs. reference)"

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K, train_properties)

    print(test_representations.shape)
    print(test_sizes)

    # Calculate prediction kernel
    Ks = get_local_kernels_gaussian(
        test_representations, train_representations, test_sizes, train_sizes, [sigma]
    )[0]

    Ks_test = np.loadtxt(ASSETS / "Ks_local_gaussian.txt")
    # Somtimes a few coulomb matrices differ because of parallel sorting and numerical error
    # Allow up to 5 molecules to differ from the supplied reference.
    differences_count = len(set(np.where(Ks - Ks_test > 1e-7)[0]))
    assert differences_count < 5, "Error in local Laplacian kernel (vs. reference)"
    # assert np.allclose(Ks, Ks_test), "Error in local Gaussian kernel (vs. reference)"

    predicted_properties = np.dot(Ks, alpha)

    mae = np.mean(np.abs(test_properties - predicted_properties))
    print(mae)
    assert abs(19.0 - mae) < 1.0, "Error in local Gaussian kernel-ridge regression"


def test_krr_laplacian_local_cmat():

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(ASSETS / "hof_qm7.txt")

    n_points = 1000

    all_representations = []
    all_properties = []
    all_atoms = []

    filenames = sorted(data.keys())[:n_points]

    for filename in filenames:
        coord, atoms = read_xyz((ASSETS / "qm7" / filename).with_suffix(".xyz"))

        representation = generate_atomic_coulomb_matrix(atoms, coord, size=23, sorting="row-norm")

        all_representations.append(representation)
        all_properties.append(data[filename])
        all_atoms.append(atoms)

    all_properties = np.array(all_properties)

    shuffle_arrays(all_atoms, all_properties, all_representations, seed=666)

    # Make training and test sets
    n_test = 100
    n_train = 200
    indices = list(range(n_points))
    train_indices = indices[:n_train]
    test_indices = indices[-n_test:]

    train_representations = np.concatenate([all_representations[i] for i in train_indices])
    test_representations = np.concatenate([all_representations[i] for i in test_indices])

    test_atoms = [all_atoms[x] for x in test_indices]
    test_properties = all_properties[test_indices]

    train_atoms = [all_atoms[x] for x in train_indices]
    train_properties = all_properties[train_indices]

    train_sizes = np.array([len(atoms) for atoms in train_atoms])
    test_sizes = np.array([len(atoms) for atoms in test_atoms])

    # Set hyper-parameters
    sigma = 10 ** (3.6)
    llambda = 10 ** (-12.0)

    K = get_local_kernels_laplacian(
        train_representations, train_representations, train_sizes, train_sizes, [sigma]
    )[0]
    assert np.allclose(K, K.T), "Error in local Laplacian kernel symmetry"

    K_test = np.loadtxt(ASSETS / "K_local_laplacian.txt")
    assert np.allclose(K, K_test), "Error in local Laplacian kernel (vs. reference)"

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K, train_properties)

    # Calculate prediction kernel
    Ks = get_local_kernels_laplacian(
        test_representations, train_representations, test_sizes, train_sizes, [sigma]
    )[0]

    Ks_test = np.loadtxt(ASSETS / "Ks_local_laplacian.txt")

    # Somtimes a few coulomb matrices differ because of parallel sorting and numerical error
    # Allow up to 5 molecules to differ from the supplied reference.
    differences_count = len(set(np.where(Ks - Ks_test > 1e-7)[0]))
    assert differences_count < 5, "Error in local Laplacian kernel (vs. reference)"

    predicted_properties = np.dot(Ks, alpha)

    mae = np.mean(np.abs(test_properties - predicted_properties))
    assert abs(8.7 - mae) < 1.0, "Error in local Laplacian kernel-ridge regression"

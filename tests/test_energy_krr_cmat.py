import numpy as np
from conftest import ASSETS, get_energies, shuffle_arrays

from qmllib.kernels import laplacian_kernel
from qmllib.representations import generate_coulomb_matrix
from qmllib.solvers import cho_solve
from qmllib.utils.xyz_format import read_xyz


def test_krr_cmat():

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(ASSETS / "hof_qm7.txt")

    n_points = 1000

    all_representations = []
    all_properties = []

    filenames = sorted(data.keys())[:n_points]

    for filename in filenames:
        coord, atoms = read_xyz((ASSETS / "qm7" / filename).with_suffix(".xyz"))

        representation = generate_coulomb_matrix(atoms, coord, size=23, sorting="row-norm")

        all_representations.append(representation)
        all_properties.append(data[filename])

    all_representations = np.array(all_representations)
    all_properties = np.array(all_properties)

    shuffle_arrays(all_properties, all_representations, seed=666)

    # Make training and test sets
    n_test = 300
    n_train = 700
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_train + n_test))

    # List of representations and properties
    test_representations = all_representations[test_indices]
    train_representations = all_representations[train_indices]
    test_properties = all_properties[test_indices]
    train_properties = all_properties[train_indices]

    # Set hyper-parameters
    sigma = 10 ** (4.2)
    llambda = 10 ** (-10.0)

    # Generate training Kernel
    K = laplacian_kernel(train_representations, train_representations, sigma)

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K, train_properties)

    # Calculate prediction kernel
    Ks = laplacian_kernel(train_representations, test_representations, sigma)
    predicted_properties = np.dot(Ks.transpose(), alpha)

    mae = np.mean(np.abs(test_properties - predicted_properties))

    assert mae < 6.0, "ERROR: Too high MAE!"

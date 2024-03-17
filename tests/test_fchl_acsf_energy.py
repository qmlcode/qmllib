import numpy as np
from conftest import ASSETS, get_energies, shuffle_arrays

from qmllib.kernels import get_local_kernel, get_local_symmetric_kernel
from qmllib.representations import generate_fchl_acsf
from qmllib.solvers import cho_solve
from qmllib.utils.xyz_format import read_xyz

np.set_printoptions(linewidth=666)


def test_energy():

    # Read the heat-of-formation energies
    data = get_energies(ASSETS / "hof_qm7.txt")

    # Generate a list
    all_representations = []
    all_properties = []
    all_atoms = []

    for xyz_file in sorted(data.keys())[:1000]:

        filename = ASSETS / "qm7" / xyz_file
        coord, atoms = read_xyz(filename)

        # Associate a property (heat of formation) with the object
        all_properties.append(data[xyz_file])

        representation = generate_fchl_acsf(atoms, coord, gradients=False, pad=27)

        all_representations.append(representation)
        all_atoms.append(atoms)

    # Convert to arrays
    all_representations = np.array(all_representations)
    all_properties = np.array(all_properties)
    # all_atoms = np.array(all_atoms)

    shuffle_arrays(all_representations, all_atoms, all_properties, seed=666)

    # Make training and test sets
    n_test = 99
    n_train = 101

    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_train + n_test))

    # List of representations
    test_representations = all_representations[test_indices]
    train_representations = all_representations[train_indices]
    test_atoms = [all_atoms[i] for i in test_indices]
    train_atoms = [all_atoms[i] for i in train_indices]
    test_properties = all_properties[test_indices]
    train_properties = all_properties[train_indices]

    # Set hyper-parameters
    sigma = 3.0
    llambda = 1e-10

    kernel = get_local_symmetric_kernel(train_representations, train_atoms, sigma)

    # Solve alpha
    alpha = cho_solve(kernel, train_properties, l2reg=llambda)

    # Calculate test kernel
    # test_kernel = get_local_kernel(train_representations, test_representations, train_atoms, test_atoms, sigma)

    # Calculate test prediction kernel
    prediction_kernel = get_local_kernel(
        train_representations, test_representations, train_atoms, test_atoms, sigma
    )
    prediction_properties = np.dot(prediction_kernel, alpha)

    mae = np.mean(np.abs(test_properties - prediction_properties))
    # assert mae < 4.0, "ERROR: Too high MAE!"
    assert mae < 4.9, "ERROR: Too high MAE!"

from pathlib import Path

import numpy as np
from conftest import ASSETS, get_energies, shuffle_arrays

from qmllib.kernels import get_local_symmetric_kernel
from qmllib.representations import generate_fchl19
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

        representation = generate_fchl19(atoms, coord, gradients=False, pad=27)

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
    all_representations[test_indices]
    train_representations = all_representations[train_indices]
    [all_atoms[i] for i in test_indices]
    train_atoms = [all_atoms[i] for i in train_indices]
    all_properties[test_indices]
    all_properties[train_indices]

    # Set hyper-parameters
    sigma = 3.0

    kernel = get_local_symmetric_kernel(train_representations, train_atoms, sigma)
    kernel_save = np.load(Path(__file__).parent / "kernel.npy")
    diff = np.abs(kernel - kernel_save)

    assert not np.any(diff > 1e-8), (
        f"Difference between original and saved kernel: max diff = {np.max(diff)}"
    )

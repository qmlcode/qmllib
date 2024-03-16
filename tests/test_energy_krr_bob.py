import numpy as np
from conftest import ASSETS, get_energies, shuffle_arrays

from qmllib.kernels import laplacian_kernel
from qmllib.representations.bob import get_asize
from qmllib.representations.representations import generate_bob
from qmllib.solvers import cho_solve
from qmllib.utils.xyz_format import read_xyz


def test_krr_bob():

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    n_points = 1000
    data = get_energies(ASSETS / "hof_qm7.txt")
    filenames = sorted(data.keys())[:n_points]

    molecules = []
    representations = []
    properties = []

    for filename in filenames:
        coord, atoms = read_xyz((ASSETS / "qm7" / filename).with_suffix(".xyz"))
        molecules.append((coord, atoms))
        properties.append(data[filename])

    size = max(atoms.size for _, atoms in molecules) + 1
    # example asize={"O": 3, "C": 7, "N": 3, "H": 16, "S": 1},
    asize = get_asize([atoms for _, atoms in molecules], 1)

    atomtypes = []
    for _, atoms in molecules:
        atomtypes.extend(atoms)
    atomtypes = np.unique(atomtypes)

    for coord, atoms in molecules:
        # representation = generate_bob(atoms, coord, atomtypes, )
        rep = generate_bob(atoms, coord, atomtypes, size=size, asize=asize)
        representations.append(rep)

    representations = np.array(representations)
    properties = np.array(properties)
    shuffle_arrays(properties, representations, seed=666)

    # Make training and test sets
    n_test = 300
    n_train = 700
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_train + n_test))

    # List of representations and properties
    test_representations = representations[test_indices]
    train_representations = representations[train_indices]
    test_properties = properties[test_indices]
    train_properties = properties[train_indices]

    # Set hyper-parameters
    sigma = 26214.40
    llambda = 1e-10

    # Generate training Kernel
    K = laplacian_kernel(train_representations, train_representations, sigma)

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K, train_properties)

    # Calculate prediction kernel
    Ks = laplacian_kernel(train_representations, test_representations, sigma)
    predicted_properties = np.dot(Ks.transpose(), alpha)

    mae = np.mean(np.abs(test_properties - predicted_properties))
    print(mae)
    assert mae < 2.6, "ERROR: Too high MAE!"

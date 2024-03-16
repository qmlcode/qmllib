import numpy as np
from conftest import ASSETS

from qmllib.representations.bob import get_asize
from qmllib.representations.representations import (
    generate_atomic_coulomb_matrix,
    generate_bob,
    generate_coulomb_matrix,
    generate_eigenvalue_coulomb_matrix,
)
from qmllib.utils.xyz_format import read_xyz


def _get_molecules():
    files = [
        ASSETS / "qm7/0101.xyz",
        ASSETS / "qm7/0102.xyz",
        ASSETS / "qm7/0103.xyz",
        ASSETS / "qm7/0104.xyz",
        ASSETS / "qm7/0105.xyz",
        ASSETS / "qm7/0106.xyz",
        ASSETS / "qm7/0107.xyz",
        ASSETS / "qm7/0108.xyz",
        ASSETS / "qm7/0109.xyz",
        ASSETS / "qm7/0110.xyz",
    ]

    mols = []
    for filename in files:
        coordinates, atoms = read_xyz(filename)
        mols.append((coordinates, atoms))

    return mols

    # size = max(atoms.size for _, atoms in mols) + 1

    # asize = get_asize([atoms for atoms in mols], 1)

    # coulomb_matrix(mols, size, path)
    # atomic_coulomb_matrix(mols, size, path)
    # eigenvalue_coulomb_matrix(mols, size, path)
    # bob(mols, size, asize, path)


def test_coulomb_matrix_rownorm():

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    # Generate coulomb matrix representation, sorted by row-norm
    representations = []
    for coordinates, nuclear_charges in mols:
        representation = generate_coulomb_matrix(
            nuclear_charges, coordinates, size=size, sorting="row-norm"
        )
        representations.append(representation)

    X_test = np.asarray([rep for rep in representations])

    print(X_test.shape)

    X_ref = np.loadtxt(ASSETS / "coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"


def test_coulomb_matrix_unsorted():

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    # Generate coulomb matrix representation, unsorted, using the Compound class
    representations = []
    for coordinates, nuclear_charges in mols:
        representation = generate_coulomb_matrix(
            nuclear_charges, coordinates, size=size, sorting="unsorted"
        )
        representations.append(representation)

    X_test = np.asarray([rep for rep in representations])

    print(X_test.shape)

    X_ref = np.loadtxt(ASSETS / "coulomb_matrix_representation_unsorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"


def test_atomic_coulomb_matrix_distance():

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    # Generate coulomb matrix representation, sorted by distance
    representations = []
    for coord, nuclear_charges in mols:
        rep = generate_atomic_coulomb_matrix(nuclear_charges, coord, size=size, sorting="distance")
        representations.append(rep)

    X_test = np.concatenate([rep for rep in representations])
    X_ref = np.loadtxt(ASSETS / "atomic_coulomb_matrix_representation_distance_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"
    # Compare to old implementation (before 'indices' keyword)
    X_ref = np.loadtxt(
        ASSETS / "atomic_coulomb_matrix_representation_distance_sorted_no_indices.txt"
    )
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"


def test_atomic_coulomb_matrix_rownorm():

    # Generate coulomb matrix representation, sorted by row-norm

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    representations = []
    for coord, nuclear_charges in mols:
        rep = generate_atomic_coulomb_matrix(nuclear_charges, coord, size=size, sorting="row-norm")
        representations.append(rep)

    X_test = np.concatenate(representations)
    X_ref = np.loadtxt(ASSETS / "atomic_coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"


def test_atomic_coulomb_matrix_distance_softcut():

    # Generate coulomb matrix representation, sorted by distance, with soft cutoffs

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    representations = []

    for coord, nuclear_charges in mols:
        rep = generate_atomic_coulomb_matrix(
            nuclear_charges,
            coord,
            size=size,
            sorting="distance",
            central_cutoff=4.0,
            central_decay=0.5,
            interaction_cutoff=5.0,
            interaction_decay=1.0,
        )
        representations.append(rep)

    X_test = np.concatenate([rep for rep in representations])
    X_ref = np.loadtxt(
        ASSETS / "atomic_coulomb_matrix_representation_distance_sorted_with_cutoff.txt"
    )
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"


def test_atomic_coulomb_matrix_rownorm_cut():

    # Generate coulomb matrix representation, sorted by row-norm, with soft cutoffs

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    representations = []

    for coord, nuclear_charges in mols:
        rep = generate_atomic_coulomb_matrix(
            nuclear_charges,
            coord,
            size=size,
            sorting="row-norm",
            central_cutoff=4.0,
            central_decay=0.5,
            interaction_cutoff=5.0,
            interaction_decay=1.0,
        )
        representations.append(rep)

    X_test = np.concatenate(representations)
    X_ref = np.loadtxt(
        ASSETS / "atomic_coulomb_matrix_representation_row-norm_sorted_with_cutoff.txt"
    )
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"


def test_atomic_coulomb_matrix_twoatom_distance():

    # Generate only two atoms in the coulomb matrix representation, sorted by distance

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    for coord, nuclear_charges in mols:
        rep = generate_atomic_coulomb_matrix(nuclear_charges, coord, size=size, sorting="distance")
        representation_subset = rep[1:3]
        rep = generate_atomic_coulomb_matrix(
            nuclear_charges, coord, size=size, sorting="distance", indices=[1, 2]
        )
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i, j] - rep[i, j]
                if abs(diff) > 1e-9:
                    print(i, j, diff, representation_subset[i, j], rep[i, j])

        assert np.allclose(
            representation_subset, rep
        ), "Error in atomic coulomb matrix representation"


def test_atomic_coulomb_matrix_twoatom_rownorm():

    # Generate only two atoms in the coulomb matrix representation, sorted by row-norm

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    for coord, nuclear_charges in mols:

        rep = generate_atomic_coulomb_matrix(nuclear_charges, coord, size=size, sorting="row-norm")
        representation_subset = rep[1:3]
        rep = generate_atomic_coulomb_matrix(
            nuclear_charges, coord, size=size, sorting="row-norm", indices=[1, 2]
        )
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i, j] - rep[i, j]
                if abs(diff) > 1e-9:
                    print(i, j, diff, representation_subset[i, j], rep[i, j])
        assert np.allclose(
            representation_subset, rep
        ), "Error in atomic coulomb matrix representation"


def test_eigenvalue_coulomb_matrix():

    # Generate coulomb matrix representation, sorted by row-norm

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    representations = []

    for coord, nuclear_charges in mols:
        rep = generate_eigenvalue_coulomb_matrix(nuclear_charges, coord, size=size)
        representations.append(rep)

    X_test = np.asarray(representations)
    X_ref = np.loadtxt(ASSETS / "eigenvalue_coulomb_matrix_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in eigenvalue coulomb matrix representation"


def test_bob():

    mols = _get_molecules()
    size = max(atoms.size for _, atoms in mols) + 1

    # example asize={"O": 3, "C": 7, "N": 3, "H": 16, "S": 1},
    asize = get_asize([atoms for _, atoms in mols], 1)

    atomtypes = []
    for _, atoms in mols:
        atomtypes.extend(atoms)
    atomtypes = np.unique(atomtypes)

    print(size)
    print(atomtypes)
    print(asize)

    representations = []

    for coord, nuclear_charges in mols:
        rep = generate_bob(nuclear_charges, coord, atomtypes, size=size, asize=asize)
        representations.append(rep)

    X_test = np.asarray(representations)
    X_ref = np.loadtxt(ASSETS / "bob_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in bag of bonds representation"

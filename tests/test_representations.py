import numpy as np
from conftest import ASSETS

from qmllib.representations.representations import generate_coulomb_matrix
from qmllib.utils.xyz_format import read_xyz


def _get_representations():
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


def test_coulomb_matrix():

    mols = _get_representations()
    size = max(atoms.size for _, atoms in mols) + 1

    # Generate coulomb matrix representation, sorted by row-norm
    representations = []
    for coordinates, nuclear_charges in mols:
        representation = generate_coulomb_matrix(
            nuclear_charges, coordinates, size=size, sorting="row-norm"
        )
        representations.append(representation)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(ASSETS / "/coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"

    # Generate coulomb matrix representation, unsorted, using the Compound class
    for coordinates, nuclear_charges in mols:
        representation = generate_coulomb_matrix(
            nuclear_charges, coordinates, size=size, sorting="unsorted"
        )
        representations.append(representation)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(ASSETS / "data/coulomb_matrix_representation_unsorted.txt")
    assert np.allclose(X_test, X_ref), "Error in coulomb matrix representation"


def test_atomic_coulomb_matrix():

    mols = _get_representations()
    size = max(atoms.size for _, atoms in mols) + 1

    # Generate coulomb matrix representation, sorted by distance
    for i, mol in enumerate(mols):
        mol.generate_atomic_coulomb_matrix(size=size, sorting="distance")

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(ASSETS / "atomic_coulomb_matrix_representation_distance_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"
    # Compare to old implementation (before 'indices' keyword)
    X_ref = np.loadtxt(
        ASSETS / "atomic_coulomb_matrix_representation_distance_sorted_no_indices.txt"
    )
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols):
        mol.generate_atomic_coulomb_matrix(size=size, sorting="row-norm")

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(ASSETS / "atomic_coulomb_matrix_representation_row-norm_sorted.txt")
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate coulomb matrix representation, sorted by distance, with soft cutoffs
    for i, mol in enumerate(mols):
        mol.generate_atomic_coulomb_matrix(
            size=size,
            sorting="distance",
            central_cutoff=4.0,
            central_decay=0.5,
            interaction_cutoff=5.0,
            interaction_decay=1.0,
        )

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(
        ASSETS / "data/atomic_coulomb_matrix_representation_distance_sorted_with_cutoff.txt"
    )
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate coulomb matrix representation, sorted by row-norm, with soft cutoffs
    for i, mol in enumerate(mols):
        mol.generate_atomic_coulomb_matrix(
            size=size,
            sorting="row-norm",
            central_cutoff=4.0,
            central_decay=0.5,
            interaction_cutoff=5.0,
            interaction_decay=1.0,
        )

    X_test = np.concatenate([mol.representation for mol in mols])
    X_ref = np.loadtxt(
        ASSETS / "data/atomic_coulomb_matrix_representation_row-norm_sorted_with_cutoff.txt"
    )
    assert np.allclose(X_test, X_ref), "Error in atomic coulomb matrix representation"

    # Generate only two atoms in the coulomb matrix representation, sorted by distance
    for i, mol in enumerate(mols):
        mol.generate_atomic_coulomb_matrix(size=size, sorting="distance")
        representation_subset = mol.representation[1:3]
        mol.generate_atomic_coulomb_matrix(size=size, sorting="distance", indices=[1, 2])
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i, j] - mol.representation[i, j]
                if abs(diff) > 1e-9:
                    print(i, j, diff, representation_subset[i, j], mol.representation[i, j])
        assert np.allclose(
            representation_subset, mol.representation
        ), "Error in atomic coulomb matrix representation"

    # Generate only two atoms in the coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols):
        mol.generate_atomic_coulomb_matrix(size=size, sorting="row-norm")
        representation_subset = mol.representation[1:3]
        mol.generate_atomic_coulomb_matrix(size=size, sorting="row-norm", indices=[1, 2])
        for i in range(2):
            for j in range(153):
                diff = representation_subset[i, j] - mol.representation[i, j]
                if abs(diff) > 1e-9:
                    print(i, j, diff, representation_subset[i, j], mol.representation[i, j])
        assert np.allclose(
            representation_subset, mol.representation
        ), "Error in atomic coulomb matrix representation"


def test_eigenvalue_coulomb_matrix(mols, size, path):

    # Generate coulomb matrix representation, sorted by row-norm
    for i, mol in enumerate(mols):
        mol.generate_eigenvalue_coulomb_matrix(size=size)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(ASSETS / "eigenvalue_coulomb_matrix_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in eigenvalue coulomb matrix representation"


def test_bob(mols, size, asize, path):

    for i, mol in enumerate(mols):
        mol.generate_bob(size=size, asize=asize)

    X_test = np.asarray([mol.representation for mol in mols])
    X_ref = np.loadtxt(ASSETS / "bob_representation.txt")
    assert np.allclose(X_test, X_ref), "Error in bag of bonds representation"


# def print_mol(mol):
#     n = len(mol.representation.shape)
#     if n == 1:
#         for item in mol.representation:
#             print("{:.9e}".format(item), end="  ")
#         print()
#     elif n == 2:
#         for atom in mol.representation:
#             for item in atom:
#                 print("{:.9e}".format(item), end="  ")
#             print()


# if __name__ == "__main__":
#     test_representations()

import numpy as np
from conftest import ASSETS

from qmllib.representations import generate_slatm, get_slatm_mbtypes
from qmllib.utils.xyz_format import read_xyz


def test_slatm_global_representation():

    files = [
        ASSETS / "qm7/0001.xyz",
        ASSETS / "qm7/0002.xyz",
        ASSETS / "qm7/0003.xyz",
        ASSETS / "qm7/0004.xyz",
        ASSETS / "qm7/0005.xyz",
        ASSETS / "qm7/0006.xyz",
        ASSETS / "qm7/0007.xyz",
        ASSETS / "qm7/0008.xyz",
        ASSETS / "qm7/0009.xyz",
        ASSETS / "qm7/0010.xyz",
    ]

    mols = []
    for xyz_file in files:
        coordinates, atoms = read_xyz(xyz_file)
        mols.append((coordinates, atoms))

    charges = [atoms for _, atoms in mols]

    mbtypes = get_slatm_mbtypes(charges)
    print("mbtypes:", mbtypes)

    representations = []
    for coord, atoms in mols:
        slatm_vector = generate_slatm(coord, atoms, mbtypes)
        representations.append(slatm_vector)

    X_qml = np.array([rep for rep in representations])
    X_ref = np.loadtxt(ASSETS / "slatm_global_representation.txt")

    assert np.allclose(X_qml, X_ref), "Error in SLATM generation"


def test_slatm_local_representation():

    files = [
        ASSETS / "qm7/0001.xyz",
        ASSETS / "qm7/0002.xyz",
        ASSETS / "qm7/0003.xyz",
        ASSETS / "qm7/0004.xyz",
        ASSETS / "qm7/0005.xyz",
        ASSETS / "qm7/0006.xyz",
        ASSETS / "qm7/0007.xyz",
        ASSETS / "qm7/0008.xyz",
        ASSETS / "qm7/0009.xyz",
        ASSETS / "qm7/0010.xyz",
    ]

    mols = []
    for xyz_file in files:
        coordinates, atoms = read_xyz(xyz_file)
        mols.append((coordinates, atoms))

    charges = [atoms for _, atoms in mols]
    mbtypes = get_slatm_mbtypes(charges)

    local_representations = []
    for _, mol in enumerate(mols):

        coord, atoms = mol
        slatm_vector = generate_slatm(coord, atoms, mbtypes, local=True)

        local_representations.append(slatm_vector)

    # Spread the structures into atom vectors
    X_qml = []
    for representation in local_representations:
        for rep in representation:
            X_qml.append(rep)

    X_qml = np.asarray(X_qml)
    X_ref = np.loadtxt(ASSETS / "slatm_local_representation.txt")

    assert np.allclose(X_qml, X_ref), "Error in SLATM generation"


if __name__ == "__main__":

    test_slatm_global_representation()
    test_slatm_local_representation()

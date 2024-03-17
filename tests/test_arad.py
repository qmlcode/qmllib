import numpy as np
from conftest import ASSETS, get_energies

from qmllib.representations.arad import (
    generate_arad_representation,
    get_atomic_kernels_arad,
    get_atomic_symmetric_kernels_arad,
    get_global_kernels_arad,
    get_global_symmetric_kernels_arad,
    get_local_kernels_arad,
    get_local_symmetric_kernels_arad,
)
from qmllib.utils.xyz_format import read_xyz


def test_arad():

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    n_points = 10
    data = get_energies(ASSETS / "hof_qm7.txt")
    filenames = sorted(data.keys())[:n_points]

    molecules = []
    representations = []
    properties = []

    for filename in filenames:
        coord, atoms = read_xyz((ASSETS / "qm7" / filename).with_suffix(".xyz"))
        molecules.append((coord, atoms))
        properties.append(data[filename])

    for coord, atoms in molecules:
        rep = generate_arad_representation(coord, atoms)
        representations.append(rep)

    representations = np.array(representations)
    properties = np.array(properties)

    # for xyz_file in sorted(data.keys())[:10]:

    #     # Initialize the qmllib.data.Compound() objects
    #     mol = qmllib.Compound(xyz=test_dir + "/qm7/" + xyz_file)

    #     # Associate a property (heat of formation) with the object
    #     mol.properties = data[xyz_file]

    #     # This is a Molecular Coulomb matrix sorted by row norm

    #     representation = generate_arad_representation(mol.coordinates, mol.nuclear_charges)

    #     mols.append(mol)

    sigmas = [25.0]

    # X1 = np.array([mol.representation for mol in mols])

    K_local_asymm = get_local_kernels_arad(representations, representations, sigmas)
    K_local_symm = get_local_symmetric_kernels_arad(representations, sigmas)

    assert np.allclose(K_local_symm, K_local_asymm), "Symmetry error in local kernels"
    assert np.invert(
        np.all(np.isnan(K_local_asymm))
    ), "ERROR: ARAD local symmetric kernel contains NaN"

    K_global_asymm = get_global_kernels_arad(representations, representations, sigmas)
    K_global_symm = get_global_symmetric_kernels_arad(representations, sigmas)

    assert np.allclose(K_global_symm, K_global_asymm), "Symmetry error in global kernels"
    assert np.invert(
        np.all(np.isnan(K_global_asymm))
    ), "ERROR: ARAD global symmetric kernel contains NaN"

    molid = 5
    coordinates, atoms = molecules[molid]
    natoms = len(atoms)
    X1 = generate_arad_representation(coordinates, atoms, size=natoms)
    XA = X1[:natoms]

    K_atomic_asymm = get_atomic_kernels_arad(XA, XA, sigmas)
    K_atomic_symm = get_atomic_symmetric_kernels_arad(XA, sigmas)

    assert np.allclose(K_atomic_symm, K_atomic_asymm), "Symmetry error in atomic kernels"
    assert np.invert(
        np.all(np.isnan(K_atomic_asymm))
    ), "ERROR: ARAD atomic symmetric kernel contains NaN"

    K_atomic_asymm = get_atomic_kernels_arad(XA, XA, sigmas)

"""
Runs the same calculation using the 'cell' keyword for generate_fchl_acsf and by straightforward
'cell cloning' done inside the test script.
"""
import os
import random
from copy import deepcopy

import numpy as np

from qmllib.representations import generate_fchl_acsf
from qmllib.utils.xyz_format import read_xyz

np.set_printoptions(linewidth=666, edgeitems=10)

REP_PARAMS = dict()
REP_PARAMS["elements"] = [1, 6, 7, 8, 16]
REP_PARAMS["rcut"] = 5.0
REP_PARAMS["acut"] = 5.0
random.seed(1)
cell_added_cutoff = 0.1


#   For given molecular coordinates generate a cell just large enough to contain the molecule.
def suitable_cell(coords):
    max_coords = None
    min_coords = None
    for atom_coords in coords:
        if max_coords is None:
            max_coords = deepcopy(atom_coords)
            min_coords = deepcopy(atom_coords)
        else:
            max_coords = np.maximum(max_coords, atom_coords)
            min_coords = np.minimum(min_coords, atom_coords)
    return np.diag((max_coords - min_coords) * (1.0 + cell_added_cutoff))


def generate_fchl_acsf_brute_pbc(nuclear_charges, coordinates, cell, gradients=False):
    num_atoms = len(nuclear_charges)
    all_coords = deepcopy(coordinates)
    all_charges = deepcopy(nuclear_charges)
    nExtend = (
        np.floor(max(REP_PARAMS["rcut"], REP_PARAMS["acut"]) / np.linalg.norm(cell, 2, axis=0)) + 1
    ).astype(int)
    print("Checked nExtend:", nExtend, ", gradient calculation:", gradients)
    for i in range(-nExtend[0], nExtend[0] + 1):
        for j in range(-nExtend[1], nExtend[1] + 1):
            for k in range(-nExtend[2], nExtend[2] + 1):
                if not (i == 0 and j == 0 and k == 0):
                    all_coords = np.append(
                        all_coords,
                        coordinates + i * cell[0, :] + j * cell[1, :] + k * cell[2, :],
                        axis=0,
                    )
                    all_charges = np.append(all_charges, nuclear_charges)
    if gradients:
        rep, drep = generate_fchl_acsf(all_charges, all_coords, gradients=True, **REP_PARAMS)
    else:
        rep = generate_fchl_acsf(all_charges, all_coords, gradients=False, **REP_PARAMS)
    rep = rep[:num_atoms, :]
    if gradients:
        drep = drep[:num_atoms, :, :num_atoms, :]
        return rep, drep
    else:
        return rep


def ragged_array_close(arr1, arr2, error_msg):
    for el1, el2 in zip(arr1, arr2):
        assert np.allclose(el1, el2), error_msg


def test_fchl_acsf_pbc():

    qm7_dir = os.path.dirname(os.path.realpath(__file__)) + "/assets/qm7"
    os.chdir(qm7_dir)
    all_xyzs = os.listdir()
    test_xyzs = random.sample(all_xyzs, 10)

    reps_no_grad1 = []
    reps_no_grad2 = []

    reps_wgrad1 = []
    reps_wgrad2 = []
    dreps1 = []
    dreps2 = []

    for xyz in test_xyzs:
        print("Tested xyz:", xyz)
        coords, atoms = read_xyz(xyz)
        cell = suitable_cell(coords)
        reps_no_grad1.append(generate_fchl_acsf_brute_pbc(atoms, coords, cell, gradients=False))
        reps_no_grad2.append(
            generate_fchl_acsf(atoms, coords, cell=cell, gradients=False, **REP_PARAMS)
        )

        rep_wgrad1, drep1 = generate_fchl_acsf_brute_pbc(atoms, coords, cell, gradients=True)
        rep_wgrad2, drep2 = generate_fchl_acsf(
            atoms, coords, cell=cell, gradients=True, **REP_PARAMS
        )

        reps_wgrad1.append(rep_wgrad1)
        reps_wgrad2.append(rep_wgrad2)

        dreps1.append(drep1)
        dreps2.append(drep2)

    ragged_array_close(
        reps_no_grad1,
        reps_no_grad2,
        "Error in PBC implementation for generate_fchl_acsf without gradients.",
    )
    ragged_array_close(
        reps_wgrad1,
        reps_wgrad2,
        "Error in PBC implementation for generate_fchl_acsf with gradients (representation).",
    )
    ragged_array_close(
        dreps1,
        dreps2,
        "Error in PBC implementation for generate_fchl_acsf with gradients (gradient of representation).",
    )
    print("Passed")


if __name__ == "__main__":

    test_fchl_acsf_pbc()

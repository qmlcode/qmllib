"""
This file contains tests for the atom centred symmetry function module.
"""

import pathlib
import random
from copy import deepcopy

import numpy as np

from qmllib.representations import generate_fchl19
from qmllib.utils.xyz_format import read_xyz

# from tests.conftest import ASSETS

np.set_printoptions(linewidth=666, edgeitems=10)
REP_PARAMS = dict()
REP_PARAMS["elements"] = [1, 6, 7, 8, 16]
REP_PARAMS["rcut"] = 8.0
REP_PARAMS["acut"] = 8.0


def pbc_corrected_drep(drep, num_atoms):
    new_shape = list(drep.shape)
    new_shape[0] = num_atoms
    new_shape[2] = num_atoms
    new_drep = np.zeros(new_shape)
    num_atoms_tot = drep.shape[0]
    for i in range(num_atoms):
        for j in range(num_atoms_tot):
            true_j = j % num_atoms
            new_drep[i, :, true_j, :] += drep[i, :, j, :]
    return new_drep


def generate_fchl19_brute_pbc(nuclear_charges, coordinates, cell, gradients=False):
    num_atoms = len(nuclear_charges)
    all_coords = deepcopy(coordinates)
    all_charges = deepcopy(nuclear_charges)
    nExtend = (
        np.floor(max(REP_PARAMS["rcut"], REP_PARAMS["acut"]) / np.linalg.norm(cell, 2, axis=0)) + 1
    ).astype(int)
    print("Checked nExtend:", nExtend)
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
        if len(all_charges) > 2500:
            return None, None
        rep, drep = generate_fchl19(all_charges, all_coords, gradients=gradients, **REP_PARAMS)
    else:
        rep = generate_fchl19(all_charges, all_coords, gradients=gradients, **REP_PARAMS)

    rep = rep[:num_atoms, :]
    if gradients:
        return rep, pbc_corrected_drep(drep, num_atoms)
    else:
        return rep


def get_acsf_numgrad(coordinates, nuclear_charges, dx=1e-6, cell=None):

    natoms = len(coordinates)
    true_coords = deepcopy(coordinates)

    true_rep = generate_fchl19(
        nuclear_charges, coordinates, gradients=False, cell=cell, **REP_PARAMS
    )

    gradient = np.zeros((3, natoms, true_rep.shape[0], true_rep.shape[1]))

    for n, coord in enumerate(true_coords):
        for xyz, x in enumerate(coord):

            temp_coords = deepcopy(true_coords)
            temp_coords[n, xyz] = x + 2.0 * dx

            rep = generate_fchl19(
                nuclear_charges, temp_coords, gradients=False, cell=cell, **REP_PARAMS
            )
            gradient[xyz, n] -= rep

            temp_coords[n, xyz] = x + dx
            rep = generate_fchl19(
                nuclear_charges, temp_coords, gradients=False, cell=cell, **REP_PARAMS
            )
            gradient[xyz, n] += 8.0 * rep

            temp_coords[n, xyz] = x - dx
            rep = generate_fchl19(
                nuclear_charges, temp_coords, gradients=False, cell=cell, **REP_PARAMS
            )
            gradient[xyz, n] -= 8.0 * rep

            temp_coords[n, xyz] = x - 2.0 * dx
            rep = generate_fchl19(
                nuclear_charges, temp_coords, gradients=False, cell=cell, **REP_PARAMS
            )
            gradient[xyz, n] += rep

    gradient /= 12 * dx

    gradient = np.swapaxes(gradient, 0, 1)
    gradient = np.swapaxes(gradient, 2, 0)
    gradient = np.swapaxes(gradient, 3, 1)

    return gradient


#   For given molecular coordinates generate a cell just large enough to contain the molecule.
def suitable_cell(coords, cell_added_cutoff=0.1):
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


def test_fchl19():

    all_xyzs = list(pathlib.Path("./assets/qm7").glob("*.xyz"))
    random.seed(1)
    xyzs = random.sample(all_xyzs, 16)
    #    xyzs=["/home/konst/qmlcode/qmllib/tests/assets/qm7/0101.xyz"]
    #    xyzs=["/home/konst/qmlcode/qmllib/tests/assets/qm7/4843.xyz"]

    for xyz in xyzs:
        print("xyz:", xyz)
        coordinates, nuclear_charges = read_xyz(xyz)

        cell = suitable_cell(coordinates)

        (repa, anal_grad) = generate_fchl19(
            nuclear_charges, coordinates, gradients=True, cell=cell, **REP_PARAMS
        )

        repb = generate_fchl19(
            nuclear_charges, coordinates, gradients=False, cell=cell, **REP_PARAMS
        )

        assert np.allclose(repa, repb), "Error in FCHL19 representation implementation"

        repc = generate_fchl19_brute_pbc(nuclear_charges, coordinates, cell)

        assert np.allclose(repa, repc), "Error in PBC implementation"

        repd, brute_pbc_grad = generate_fchl19_brute_pbc(
            nuclear_charges, coordinates, cell, gradients=True
        )
        if repd is None:
            print("too large gradient matrix for brute PBC check")
        else:
            assert np.allclose(repd, repa)
            assert np.allclose(
                anal_grad, brute_pbc_grad
            ), "Error in FCHL-ACSF gradient implementation"

        num_grad = get_acsf_numgrad(coordinates, nuclear_charges, cell=cell)
        print(
            "analytic-numerical gradient difference vs. average magnitude:",
            np.max(np.abs(num_grad - anal_grad)),
            np.mean(np.abs(num_grad)),
        )


#        assert np.allclose(anal_grad, brute_pbc_grad), "Error in FCHL-ACSF gradient implementation"


test_fchl19()

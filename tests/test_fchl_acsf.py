"""
This file contains tests for the atom centred symmetry function module.
"""

from copy import deepcopy

import numpy as np
from conftest import ASSETS

from qmllib.representations import generate_fchl_acsf
from qmllib.utils.xyz_format import read_xyz

np.set_printoptions(linewidth=666, edgeitems=10)
REP_PARAMS = dict()
REP_PARAMS["elements"] = [1, 6, 7]


def get_acsf_numgrad(coordinates, nuclear_charges, dx=1e-5):

    natoms = len(coordinates)
    true_coords = deepcopy(coordinates)

    true_rep = generate_fchl_acsf(nuclear_charges, coordinates, gradients=False, **REP_PARAMS)

    gradient = np.zeros((3, natoms, true_rep.shape[0], true_rep.shape[1]))

    for n, coord in enumerate(true_coords):
        for xyz, x in enumerate(coord):

            temp_coords = deepcopy(true_coords)
            temp_coords[n, xyz] = x + 2.0 * dx

            (rep, grad) = generate_fchl_acsf(
                nuclear_charges, temp_coords, gradients=True, **REP_PARAMS
            )
            gradient[xyz, n] -= rep

            temp_coords[n, xyz] = x + dx
            (rep, grad) = generate_fchl_acsf(
                nuclear_charges, temp_coords, gradients=True, **REP_PARAMS
            )
            gradient[xyz, n] += 8.0 * rep

            temp_coords[n, xyz] = x - dx
            (rep, grad) = generate_fchl_acsf(
                nuclear_charges, temp_coords, gradients=True, **REP_PARAMS
            )
            gradient[xyz, n] -= 8.0 * rep

            temp_coords[n, xyz] = x - 2.0 * dx
            (rep, grad) = generate_fchl_acsf(
                nuclear_charges, temp_coords, gradients=True, **REP_PARAMS
            )
            gradient[xyz, n] += rep

    gradient /= 12 * dx

    gradient = np.swapaxes(gradient, 0, 1)
    gradient = np.swapaxes(gradient, 2, 0)
    gradient = np.swapaxes(gradient, 3, 1)

    return gradient


def test_fchl_acsf():

    coordinates, nuclear_charges = read_xyz(ASSETS / "qm7/0101.xyz")

    (repa, anal_grad) = generate_fchl_acsf(
        nuclear_charges, coordinates, gradients=True, **REP_PARAMS
    )

    repb = generate_fchl_acsf(nuclear_charges, coordinates, gradients=False, **REP_PARAMS)

    assert np.allclose(repa, repb), "Error in FCHL-ACSF representation implementation"

    num_grad = get_acsf_numgrad(coordinates, nuclear_charges)

    assert np.allclose(anal_grad, num_grad), "Error in FCHL-ACSF gradient implementation"

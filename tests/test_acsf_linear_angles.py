
#

#






#


#








"""
This file contains tests for the atom centred symmetry function module.
"""
from __future__ import print_function
import os
from copy import deepcopy
import numpy as np
np.set_printoptions(linewidth=666, edgeitems=100000000000000000)
import qmllib
from qmllib import Compound
from qmllib.representations import generate_fchl_acsf
from qmllib.representations import generate_acsf

REP_PARAMS = dict()
REP_PARAMS["elements"] = [1, 6, 7]
# REP_PARAMS["pad"] = 
# REP_PARAMS["nRs2"] = 30
# REP_PARAMS["nRs3"] = 3

def get_fchl_acsf_numgrad(mol, dx=1e-5):

    true_coords = deepcopy(mol.coordinates)

    true_rep = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=False, **REP_PARAMS)

    gradient = np.zeros((3, mol.natoms, true_rep.shape[0], true_rep.shape[1]))

    for n, coord in enumerate(true_coords):
        for xyz, x in enumerate(coord):

            temp_coords = deepcopy(true_coords)
            temp_coords[n,xyz] = x + 2.0 *dx

            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] -= rep
            
            temp_coords[n,xyz] = x + dx
            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] += 8.0 * rep
            
            temp_coords[n,xyz] = x - dx
            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] -= 8.0 * rep
            
            temp_coords[n,xyz] = x - 2.0 *dx
            (rep, grad) = generate_fchl_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] += rep

    gradient /= (12 * dx)

    gradient = np.swapaxes(gradient, 0, 1 )
    gradient = np.swapaxes(gradient, 2, 0 )
    gradient = np.swapaxes(gradient, 3, 1)

    return gradient


def get_acsf_numgrad(mol, dx=1e-5):

    true_coords = deepcopy(mol.coordinates)

    true_rep = generate_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=False, **REP_PARAMS)

    gradient = np.zeros((3, mol.natoms, true_rep.shape[0], true_rep.shape[1]))

    for n, coord in enumerate(true_coords):
        for xyz, x in enumerate(coord):

            temp_coords = deepcopy(true_coords)
            temp_coords[n,xyz] = x + 2.0 *dx

            (rep, grad) = generate_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] -= rep
            
            temp_coords[n,xyz] = x + dx
            (rep, grad) = generate_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] += 8.0 * rep
            
            temp_coords[n,xyz] = x - dx
            (rep, grad) = generate_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] -= 8.0 * rep
            
            temp_coords[n,xyz] = x - 2.0 *dx
            (rep, grad) = generate_acsf(mol.nuclear_charges, temp_coords, gradients=True, **REP_PARAMS)
            gradient[xyz, n] += rep

    gradient /= (12 * dx)

    gradient = np.swapaxes(gradient, 0, 1 )
    gradient = np.swapaxes(gradient, 2, 0 )
    gradient = np.swapaxes(gradient, 3, 1)

    return gradient

    
def test_fchl_acsf():
    
    test_dir = os.path.dirname(os.path.realpath(__file__))

    # mol = Compound(xyz=test_dir+ "/qm7/0101.xyz")
    mol = Compound(xyz=test_dir+ "/data/hcn.xyz")

    (repa, anal_grad) = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=True,  **REP_PARAMS)
  
    # help(generate_fchl_acsf)
    print("ANALYTICAL")
    print(anal_grad[0])

    repb = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=False, **REP_PARAMS)

    assert np.allclose(repa, repb), "Error in FCHL-ACSF representation implementation"

    num_grad = get_fchl_acsf_numgrad(mol)
    
    print("NUMERICAL")
    print(num_grad[0])

    assert np.allclose(anal_grad, num_grad), "Error in FCHL-ACSF gradient implementation"

    
def test_acsf():
    
    test_dir = os.path.dirname(os.path.realpath(__file__))

    # mol = Compound(xyz=test_dir+ "/qm7/0101.xyz")
    mol = Compound(xyz=test_dir+ "/data/hcn.xyz")

    (repa, anal_grad) = generate_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=True,  **REP_PARAMS)
  
    # help(generate_fchl_acsf)
    print("ANALYTICAL")
    # print(anal_grad[0])

    repb = generate_acsf(mol.nuclear_charges, mol.coordinates, 
        gradients=False, **REP_PARAMS)

    assert np.allclose(repa, repb), "Error in FCHL-ACSF representation implementation"

    num_grad = get_acsf_numgrad(mol)
    
    print("NUMERICAL")
    # print(num_grad[0])

    assert np.allclose(anal_grad, num_grad), "Error in FCHL-ACSF gradient implementation"

if __name__ == "__main__":

    test_fchl_acsf()
    test_acsf()


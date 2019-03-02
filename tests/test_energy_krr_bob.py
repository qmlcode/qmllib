
#

#






#


#








from __future__ import print_function

import os
import numpy as np

import qmllib

from qmllib.kernels import laplacian_kernel
from qmllib.math import cho_solve
from qmllib.representations import get_slatm_mbtypes


def get_energies(filename):
    """ Returns a dictionary with heats of formation for each xyz-file.
    """

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])

        energies[xyz_name] = hof

    return energies

def test_krr_bob():

    test_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
    data = get_energies(test_dir + "/data/hof_qm7.txt")

    # Generate a list of qmllib.data.Compound() objects
    mols = []

    for xyz_file in sorted(data.keys())[:1000]:

        # Initialize the qmllib.Compound() objects
        mol = qmllib.Compound(xyz=test_dir + "/qm7/" + xyz_file)

        # Associate a property (heat of formation) with the object
        mol.properties = data[xyz_file]

        # This is a Molecular Coulomb matrix sorted by row norm
        mol.generate_bob()

        mols.append(mol)

    # Shuffle molecules
    np.random.seed(666)
    np.random.shuffle(mols)

    # Make training and test sets
    n_test  = 300
    n_train = 700

    training = mols[:n_train]
    test  = mols[-n_test:]

    # List of representations
    X  = np.array([mol.representation for mol in training])
    Xs = np.array([mol.representation for mol in test])

    # List of properties
    Y = np.array([mol.properties for mol in training])
    Ys = np.array([mol.properties for mol in test])

    # Set hyper-parameters
    sigma = 26214.40
    llambda = 1e-10

    # Generate training Kernel
    K = laplacian_kernel(X, X, sigma)

    # Solve alpha
    K[np.diag_indices_from(K)] += llambda
    alpha = cho_solve(K,Y)

    # Calculate prediction kernel
    Ks = laplacian_kernel(X, Xs, sigma)
    Yss = np.dot(Ks.transpose(), alpha)

    mae = np.mean(np.abs(Ys - Yss))
    print(mae)
    assert mae < 2.6, "ERROR: Too high MAE!"

if __name__ == "__main__":

    test_krr_bob()

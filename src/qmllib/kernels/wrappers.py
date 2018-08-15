
#

#






#


#








import numpy as np

from .fkernels import fget_vector_kernels_gaussian
from .fkernels import fget_vector_kernels_laplacian
from .fkernels import fget_local_kernels_gaussian

from ..arad import get_local_kernels_arad
from ..arad import get_local_symmetric_kernels_arad


def get_atomic_kernels_laplacian(mols1, mols2, sigmas):

    n1 = np.array([mol.natoms for mol in mols1], dtype=np.int32)
    n2 = np.array([mol.natoms for mol in mols2], dtype=np.int32)

    max1 = np.max(n1)
    max2 = np.max(n2)

    nm1 = n1.size
    nm2 = n2.size

    cmat_size = mols1[0].representation.shape[1]

    x1 = np.zeros((nm1, max1, cmat_size), dtype=np.float64, order="F")
    x2 = np.zeros((nm2, max2, cmat_size), dtype=np.float64, order="F")

    for imol in range(nm1):
        x1[imol,:n1[imol],:cmat_size] = mols1[imol].representation

    for imol in range(nm2):
        x2[imol,:n2[imol],:cmat_size] = mols2[imol].representation

    # Reorder for Fortran speed
    x1 = np.swapaxes(x1, 0, 2)
    x2 = np.swapaxes(x2, 0, 2)

    sigmas = np.asarray(sigmas, dtype=np.float64)
    nsigmas = sigmas.size

    return fget_vector_kernels_laplacian(x1, x2, n1, n2, sigmas, 
        nm1, nm2, nsigmas)


def get_atomic_kernels_gaussian(mols1, mols2, sigmas):

    n1 = np.array([mol.natoms for mol in mols1], dtype=np.int32)
    n2 = np.array([mol.natoms for mol in mols2], dtype=np.int32)

    max1 = np.max(n1)
    max2 = np.max(n2)

    nm1 = n1.size
    nm2 = n2.size

    cmat_size = mols1[0].representation.shape[1]

    x1 = np.zeros((nm1, max1, cmat_size), dtype=np.float64, order="F")
    x2 = np.zeros((nm2, max2, cmat_size), dtype=np.float64, order="F")

    for imol in range(nm1):
        x1[imol,:n1[imol],:cmat_size] = mols1[imol].representation

    for imol in range(nm2):
        x2[imol,:n2[imol],:cmat_size] = mols2[imol].representation

    # Reorder for Fortran speed
    x1 = np.swapaxes(x1, 0, 2)
    x2 = np.swapaxes(x2, 0, 2)

    sigmas = np.array(sigmas, dtype=np.float64)
    nsigmas = sigmas.size

    return fget_vector_kernels_gaussian(x1, x2, n1, n2, sigmas, 
        nm1, nm2, nsigmas)


def arad_local_kernels(mols1, mols2, sigmas,
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):

    amax = mols1[0].representation.shape[0]

    nm1 = len(mols1)
    nm2 = len(mols2)

    X1 = np.array([mol.representation for mol in mols1]).reshape((nm1, amax, 5, amax))
    X2 = np.array([mol.representation for mol in mols2]).reshape((nm2, amax, 5, amax))

    K = get_local_kernels_arad(X1, X2, sigmas, 
        width=width, cut_distance=cut_distance, r_width=r_width, c_width=c_width)

    return K


def arad_local_symmetric_kernels(mols1, sigmas,
        width=0.2, cut_distance=5.0, r_width=1.0, c_width=0.5):

    amax = mols1[0].representation.shape[0]
    nm1 = len(mols1)

    X1 = np.array([mol.representation for mol in mols1]).reshape((nm1,amax,5,amax))

    K = get_local_symmetric_kernels_arad(X1, sigmas, \
        width=width, cut_distance=cut_distance, r_width=r_width, c_width=c_width)

    return K

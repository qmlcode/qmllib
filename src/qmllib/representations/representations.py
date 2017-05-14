
#

#






#


#








import numpy as np

from .frepresentations import fgenerate_coulomb_matrix
from .frepresentations import fgenerate_unsorted_coulomb_matrix
from .frepresentations import fgenerate_local_coulomb_matrix
from .frepresentations import fgenerate_atomic_coulomb_matrix


def generate_coulomb_matrix(coordinates, nuclear_charges, size=23, sorting="row-norm"):

    if (sorting == "row-norm"):
        return fgenerate_coulomb_matrix(nuclear_charges, \
            coordinates, len(nuclear_charges), size)

    elif (sorting == "unsorted"):
        return fgenerate_unsorted_coulomb_matrix(nuclear_charges, \
            coordinates, len(nuclear_charges), size)

    else:
        print "ERROR: Unknown sorting scheme requested"


def generate_atomic_coulomb_matrix(self,size=23, sorting ="row-norm"):

    if (sorting == "row-norm"):
        self.local_coulomb_matrix = fgenerate_local_coulomb_matrix( \
            self.nuclear_charges, self.coordinates, self.natoms, size)

    elif (sorting == "distance"):
        self.atomic_coulomb_matrix = fgenerate_atomic_coulomb_matrix( \
            self.nuclear_charges, self.coordinates, self.natoms, size)

    else:
        print "ERROR: Unknown sorting scheme requested"

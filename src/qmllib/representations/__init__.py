
#

#






#


#








"""
FML representation
=====
Provides
  1. coulomb_matrix
  1. local_coulomb_matrix
"""

from .frepresentations import fgenerate_coulomb_matrix
from .frepresentations import fgenerate_unsorted_coulomb_matrix
from .frepresentations import fgenerate_local_coulomb_matrix
from .frepresentations import fgenerate_atomic_coulomb_matrix

from .representations import generate_coulomb_matrix
from .representations import generate_atomic_coulomb_matrix

from .arad import ARAD

__all__ = ['generate_coulomb_matrix', 'generate_atomic_coulomb_matrix']


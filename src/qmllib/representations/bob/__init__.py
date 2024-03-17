"""
Bag of bonds utils functions
"""

from typing import List

import numpy as np
from numpy import ndarray

from qmllib.constants.periodic_table import ELEMENT_NAME


def get_natypes(nuclear_charges: np.ndarray) -> dict[str, int]:
    """Get number of atom types, from a list of molecules"""

    keys, counts = np.unique(nuclear_charges, return_counts=True)

    # natypes = dict([(key, len(value)) for key,value in self.atomtype_indices.items()])

    # natypes = dict([ (key, count) for key, count in zip(keys, counts)])

    keys_name = [ELEMENT_NAME[key] for key in keys]

    natypes = dict([(key, count) for key, count in zip(keys_name, counts)])

    return natypes


def get_asize(list_nuclear_charges: List[ndarray], pad: int) -> dict[str, int]:
    """

    example:
        asize = {"O":3, "C":7, "N":3, "H":16, "S":1}
    """

    asize: dict[str, int] = dict()

    for nuclear_charges in list_nuclear_charges:
        natypes = get_natypes(nuclear_charges)
        for key, value in natypes.items():
            try:
                asize[key] = max(asize[key], value + pad)
            except KeyError:
                asize[key] = value + pad
    return asize

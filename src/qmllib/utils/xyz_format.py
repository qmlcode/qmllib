from pathlib import Path
from typing import Tuple

import numpy as np
from numpy import ndarray

from qmllib.constants.periodic_table import NUCLEAR_CHARGE


def read_xyz(filename: str | Path) -> Tuple[ndarray, ndarray]:
    """(Re-)initializes the Compound-object with data from an xyz-file.

    :param filename: Input xyz-filename or file-like obejct
    :type filename: string or file-like object
    """

    # if isinstance(filename, string_types):
    #     with open(filename, "r") as f:
    #         lines = f.readlines()
    # else:
    #     lines = filename.readlines()

    with open(filename, "r") as f:
        lines = f.readlines()

    natoms = int(lines[0])
    atomtypes = []
    nuclear_charges = np.empty(natoms, dtype=int)
    coordinates = np.empty((natoms, 3), dtype=float)

    # Give the Compound a name if it is a string
    # name = filename if isinstance(filename, string_types) else "Compound"

    for i, line in enumerate(lines[2 : natoms + 2]):
        tokens = line.split()

        if len(tokens) < 4:
            break

        atomtypes.append(tokens[0])
        nuclear_charges[i] = NUCLEAR_CHARGE[tokens[0]]

        coordinates[i] = np.asarray(tokens[1:4], dtype=float)

    return coordinates, nuclear_charges


# self.natypes = dict([(key, len(value)) for key,value in self.atomtype_indices.items()])

from pathlib import Path

import numpy as np

ASSETS = Path("./tests/assets")


def get_asize(list_of_atoms, pad):
    """TODO Anders what is asize"""

    asize: dict[int, int] = dict()

    # WHAT

    for atoms in list_of_atoms:

        unique_atoms, unique_counts = np.unique(atoms, return_counts=True)

        for atom, count in zip(unique_atoms, unique_counts):

            prev = asize.get(atom, None)

            if prev is None:
                asize[atom] = count + pad
                continue

            asize[atom] = max(asize[atom], count + pad)

        # for key, value in mol.natypes.items():
        #     try:
        #         asize[key] = max(asize[key], value + pad)
        #     except KeyError:
        #         asize[key] = value + pad

    return asize


def get_energies(filename: Path):
    """Returns a dictionary with heats of formation for each xyz-file."""

    with open(filename, "r") as f:
        lines = f.readlines()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])

        energies[xyz_name] = hof

    return energies

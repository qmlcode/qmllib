from __future__ import print_function

import os

from qmllib import Compound


def compare_lists(a, b):
    for pair in zip(a, b):
        if pair[0] != pair[1]:
            return False
    return True


def test_compound():

    test_dir = os.path.dirname(os.path.realpath(__file__))
    c = Compound(xyz=test_dir + "/data/compound_test.xyz")

    ref_atomtypes = ["C", "Cl", "Br", "H", "H"]
    ref_charges = [6, 17, 35, 1, 1]

    assert compare_lists(ref_atomtypes, c.atomtypes), "Failed parsing atomtypes"
    assert compare_lists(ref_charges, c.nuclear_charges), "Failed parsing nuclear_charges"

    # Test extended xyz
    c2 = Compound(xyz=test_dir + "/data/compound_test.exyz")

    ref_atomtypes = ["C", "Cl", "Br", "H", "H"]
    ref_charges = [6, 17, 35, 1, 1]

    assert compare_lists(ref_atomtypes, c2.atomtypes), "Failed parsing atomtypes"
    assert compare_lists(ref_charges, c2.nuclear_charges), "Failed parsing nuclear_charges"

    # Test extended xyz from a file pointer rather than a filename
    with open(test_dir + "/data/compound_test.exyz") as fp:
        c3 = Compound(xyz=fp)

    assert compare_lists(ref_atomtypes, c3.atomtypes), "Failed parsing atomtypes"
    assert compare_lists(ref_charges, c3.nuclear_charges), "Failed parsing nuclear_charges"


if __name__ == "__main__":

    test_compound()
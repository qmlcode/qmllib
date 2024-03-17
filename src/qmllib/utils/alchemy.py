from copy import copy
from typing import Any, Dict, Tuple, Union

import numpy as np
from numpy import float64, ndarray

# Periodic table indexes
PTP = {
    1: [1, 1],
    2: [1, 8],  # Row1
    3: [2, 1],
    4: [2, 2],  # Row2\
    5: [2, 3],
    6: [2, 4],
    7: [2, 5],
    8: [2, 6],
    9: [2, 7],
    10: [2, 8],
    11: [3, 1],
    12: [3, 2],  # Row3\
    13: [3, 3],
    14: [3, 4],
    15: [3, 5],
    16: [3, 6],
    17: [3, 7],
    18: [3, 8],
    19: [4, 1],
    20: [4, 2],  # Row4\
    31: [4, 3],
    32: [4, 4],
    33: [4, 5],
    34: [4, 6],
    35: [4, 7],
    36: [4, 8],
    21: [4, 9],
    22: [4, 10],
    23: [4, 11],
    24: [4, 12],
    25: [4, 13],
    26: [4, 14],
    27: [4, 15],
    28: [4, 16],
    29: [4, 17],
    30: [4, 18],
    37: [5, 1],
    38: [5, 2],  # Row5\
    49: [5, 3],
    50: [5, 4],
    51: [5, 5],
    52: [5, 6],
    53: [5, 7],
    54: [5, 8],
    39: [5, 9],
    40: [5, 10],
    41: [5, 11],
    42: [5, 12],
    43: [5, 13],
    44: [5, 14],
    45: [5, 15],
    46: [5, 16],
    47: [5, 17],
    48: [5, 18],
    55: [6, 1],
    56: [6, 2],  # Row6\
    81: [6, 3],
    82: [6, 4],
    83: [6, 5],
    84: [6, 6],
    85: [6, 7],
    86: [6, 8],
    72: [6, 10],
    73: [6, 11],
    74: [6, 12],
    75: [6, 13],
    76: [6, 14],
    77: [6, 15],
    78: [6, 16],
    79: [6, 17],
    80: [6, 18],
    57: [6, 19],
    58: [6, 20],
    59: [6, 21],
    60: [6, 22],
    61: [6, 23],
    62: [6, 24],
    63: [6, 25],
    64: [6, 26],
    65: [6, 27],
    66: [6, 28],
    67: [6, 29],
    68: [6, 30],
    69: [6, 31],
    70: [6, 32],
    71: [6, 33],
    87: [7, 1],
    88: [7, 2],  # Row7\
    113: [7, 3],
    114: [7, 4],
    115: [7, 5],
    116: [7, 6],
    117: [7, 7],
    118: [7, 8],
    104: [7, 10],
    105: [7, 11],
    106: [7, 12],
    107: [7, 13],
    108: [7, 14],
    109: [7, 15],
    110: [7, 16],
    111: [7, 17],
    112: [7, 18],
    89: [7, 19],
    90: [7, 20],
    91: [7, 21],
    92: [7, 22],
    93: [7, 23],
    94: [7, 24],
    95: [7, 25],
    96: [7, 26],
    97: [7, 27],
    98: [7, 28],
    99: [7, 29],
    100: [7, 30],
    101: [7, 32],
    102: [7, 14],
    103: [7, 33],
}

QtNm = {
    # Row1
    1: [1, 0, 0, 1.0 / 2.0],
    2: [1, 0, 0, -1.0 / 2.0]
    # Row2
    ,
    3: [2, 0, 0, 1.0 / 2.0],
    4: [2, 0, 0, -1.0 / 2.0],
    5: [2, -1, 1, 1.0 / 2.0],
    6: [2, 0, 1, 1.0 / 2.0],
    7: [2, 1, 1, 1.0 / 2.0],
    8: [2, -1, 1, -1.0 / 2.0],
    9: [2, 0, 1, -1.0 / 2.0],
    10: [2, 1, 1, -1.0 / 2.0]
    # Row3
    ,
    11: [3, 0, 0, 1.0 / 2.0],
    12: [3, 0, 0, -1.0 / 2.0],
    13: [3, -1, 1, 1.0 / 2.0],
    14: [3, 0, 1, 1.0 / 2.0],
    15: [3, 1, 1, 1.0 / 2.0],
    16: [3, -1, 1, -1.0 / 2.0],
    17: [3, 0, 1, -1.0 / 2.0],
    18: [3, 1, 1, -1.0 / 2.0]
    # Row3
    ,
    19: [4, 0, 0, 1.0 / 2.0],
    20: [4, 0, 0, -1.0 / 2.0],
    31: [4, -1, 2, 1.0 / 2.0],
    32: [4, 0, 1, 1.0 / 2.0],
    33: [4, 1, 1, 1.0 / 2.0],
    34: [4, -1, 1, -1.0 / 2.0],
    35: [4, 0, 1, -1.0 / 2.0],
    36: [4, 1, 1, -1.0 / 2.0],
    21: [4, -2, 2, 1.0 / 2.0],
    22: [4, -1, 2, 1.0 / 2.0],
    23: [4, 0, 2, 1.0 / 2.0],
    24: [4, 1, 2, 1.0 / 2.0],
    25: [4, 2, 2, 1.0 / 2.0],
    26: [4, -2, 2, -1.0 / 2.0],
    27: [4, -1, 2, -1.0 / 2.0],
    28: [4, 0, 2, -1.0 / 2.0],
    29: [4, 1, 2, -1.0 / 2.0],
    30: [4, 2, 2, -1.0 / 2.0]
    # Row5
    ,
    37: [5, 0, 0, 1.0 / 2.0],
    38: [5, 0, 0, -1.0 / 2.0],
    49: [5, -1, 1, 1.0 / 2.0],
    50: [5, 0, 1, 1.0 / 2.0],
    51: [5, 1, 1, 1.0 / 2.0],
    52: [5, -1, 1, -1.0 / 2.0],
    53: [5, 0, 1, -1.0 / 2.0],
    54: [5, 1, 1, -1.0 / 2.0],
    39: [5, -2, 2, 1.0 / 2.0],
    40: [5, -1, 2, 1.0 / 2.0],
    41: [5, 0, 2, 1.0 / 2.0],
    42: [5, 1, 2, 1.0 / 2.0],
    43: [5, 2, 2, 1.0 / 2.0],
    44: [5, -2, 2, -1.0 / 2.0],
    45: [5, -1, 2, -1.0 / 2.0],
    46: [5, 0, 2, -1.0 / 2.0],
    47: [5, 1, 2, -1.0 / 2.0],
    48: [5, 2, 2, -1.0 / 2.0]
    # Row6
    ,
    55: [6, 0, 0, 1.0 / 2.0],
    56: [6, 0, 0, -1.0 / 2.0],
    81: [6, -1, 1, 1.0 / 2.0],
    82: [6, 0, 1, 1.0 / 2.0],
    83: [6, 1, 1, 1.0 / 2.0],
    84: [6, -1, 1, -1.0 / 2.0],
    85: [6, 0, 1, -1.0 / 2.0],
    86: [6, 1, 1, -1.0 / 2.0],
    71: [6, -2, 2, 1.0 / 2.0],
    72: [6, -1, 2, 1.0 / 2.0],
    73: [6, 0, 2, 1.0 / 2.0],
    74: [6, 1, 2, 1.0 / 2.0],
    75: [6, 2, 2, 1.0 / 2.0],
    76: [6, -2, 2, -1.0 / 2.0],
    77: [6, -1, 2, -1.0 / 2.0],
    78: [6, 0, 2, -1.0 / 2.0],
    79: [6, 1, 2, -1.0 / 2.0],
    80: [6, 2, 2, -1.0 / 2.0],
    57: [6, -3, 3, 1.0 / 2.0],
    58: [6, -2, 3, 1.0 / 2.0],
    59: [6, -1, 3, 1.0 / 2.0],
    60: [6, 0, 3, 1.0 / 2.0],
    61: [6, 1, 3, 1.0 / 2.0],
    62: [6, 2, 3, 1.0 / 2.0],
    63: [6, 3, 3, 1.0 / 2.0],
    64: [6, -3, 3, -1.0 / 2.0],
    65: [6, -2, 3, -1.0 / 2.0],
    66: [6, -1, 3, -1.0 / 2.0],
    67: [6, 0, 3, -1.0 / 2.0],
    68: [6, 1, 3, -1.0 / 2.0],
    69: [6, 2, 3, -1.0 / 2.0],
    70: [6, 3, 3, -1.0 / 2.0]
    # Row7
    ,
    87: [7, 0, 0, 1.0 / 2.0],
    88: [7, 0, 0, -1.0 / 2.0],
    113: [7, -1, 1, 1.0 / 2.0],
    114: [7, 0, 1, 1.0 / 2.0],
    115: [7, 1, 1, 1.0 / 2.0],
    116: [7, -1, 1, -1.0 / 2.0],
    117: [7, 0, 1, -1.0 / 2.0],
    118: [7, 1, 1, -1.0 / 2.0],
    103: [7, -2, 2, 1.0 / 2.0],
    104: [7, -1, 2, 1.0 / 2.0],
    105: [7, 0, 2, 1.0 / 2.0],
    106: [7, 1, 2, 1.0 / 2.0],
    107: [7, 2, 2, 1.0 / 2.0],
    108: [7, -2, 2, -1.0 / 2.0],
    109: [7, -1, 2, -1.0 / 2.0],
    110: [7, 0, 2, -1.0 / 2.0],
    111: [7, 1, 2, -1.0 / 2.0],
    112: [7, 2, 2, -1.0 / 2.0],
    89: [7, -3, 3, 1.0 / 2.0],
    90: [7, -2, 3, 1.0 / 2.0],
    91: [7, -1, 3, 1.0 / 2.0],
    92: [7, 0, 3, 1.0 / 2.0],
    93: [7, 1, 3, 1.0 / 2.0],
    94: [7, 2, 3, 1.0 / 2.0],
    95: [7, 3, 3, 1.0 / 2.0],
    96: [7, -3, 3, -1.0 / 2.0],
    97: [7, -2, 3, -1.0 / 2.0],
    98: [7, -1, 3, -1.0 / 2.0],
    99: [7, 0, 3, -1.0 / 2.0],
    100: [7, 1, 3, -1.0 / 2.0],
    101: [7, 2, 3, -1.0 / 2.0],
    102: [7, 3, 3, -1.0 / 2.0],
}


def get_alchemy(
    alchemy: Union[ndarray, str],
    emax: int = 100,
    r_width: float = 0.001,
    c_width: float = 0.001,
    elemental_vectors: Dict[Any, Any] = {},
    n_width: float = 0.001,
    m_width: float = 0.001,
    l_width: float = 0.001,
    s_width: float = 0.001,
) -> Tuple[bool, ndarray]:

    if isinstance(alchemy, np.ndarray):

        doalchemy = True
        return doalchemy, alchemy

    elif alchemy == "off":

        pd = np.eye(emax)
        doalchemy = False

        return doalchemy, pd

    elif alchemy == "periodic-table":

        pd = gen_pd(emax=emax, r_width=r_width, c_width=c_width)
        doalchemy = True

        return doalchemy, pd

    elif alchemy == "quantum-numbers":
        pd = gen_QNum_distances(
            emax=emax, n_width=n_width, m_width=m_width, l_width=l_width, s_width=s_width
        )
        doalchemy = True

        return doalchemy, pd

    elif alchemy == "custom":

        pd = gen_custom(elemental_vectors, emax)
        doalchemy = True

        return doalchemy, pd

    raise NotImplementedError("QML ERROR: Unknown alchemical method specified:", alchemy)


def QNum_distance(a, b, n_width, m_width, l_width, s_width):
    """Calculate stochiometric distance
    a -- nuclear charge of element a
    b -- nuclear charge of element b
    r_width -- sigma in row-direction
    c_width -- sigma in column direction
    """

    na = QtNm[int(a)][0]
    nb = QtNm[int(b)][0]

    ma = QtNm[int(a)][1]
    mb = QtNm[int(b)][1]

    la = QtNm[int(a)][2]
    lb = QtNm[int(b)][2]

    sa = QtNm[int(a)][3]
    sb = QtNm[int(b)][3]

    return np.exp(
        -((na - nb) ** 2) / (4 * n_width**2)
        - (ma - mb) ** 2 / (4 * m_width**2)
        - (la - lb) ** 2 / (4 * l_width**2)
        - (sa - sb) ** 2 / (4 * s_width**2)
    )


def gen_QNum_distances(emax=100, n_width=0.001, m_width=0.001, l_width=0.001, s_width=0.001):
    """Generate stochiometric ditance matrix
    emax -- Largest element
    r_width -- sigma in row-direction
    c_width -- sigma in column direction
    """

    pd = np.zeros((emax, emax))

    for i in range(emax):
        for j in range(emax):

            pd[i, j] = QNum_distance(i + 1, j + 1, n_width, m_width, l_width, s_width)

    return pd


def periodic_distance(a: int, b: int, r_width: float, c_width: float) -> float64:
    """Calculate stochiometric distance

    a -- nuclear charge of element a
    b -- nuclear charge of element b
    r_width -- sigma in row-direction
    c_width -- sigma in column direction
    """

    ra = PTP[int(a)][0]
    rb = PTP[int(b)][0]
    ca = PTP[int(a)][1]
    cb = PTP[int(b)][1]

    # return (r_width**2 * c_width**2) / ((r_width**2 + (ra - rb)**2) * (c_width**2 + (ca - cb)**2))

    return np.exp(-((ra - rb) ** 2) / (4 * r_width**2) - (ca - cb) ** 2 / (4 * c_width**2))


def gen_pd(emax: int = 100, r_width: float = 0.001, c_width: float = 0.001) -> ndarray:
    """Generate stochiometric ditance matrix

    emax -- Largest element
    r_width -- sigma in row-direction
    c_width -- sigma in column direction
    """

    pd = np.zeros((emax, emax))

    for i in range(emax):
        for j in range(emax):
            pd[i, j] = periodic_distance(i + 1, j + 1, r_width, c_width)

    return pd


def gen_custom(e_vec, emax=100):
    """Generate stochiometric ditance matrix
    emax -- Largest element
    r_width -- sigma in row-direction
    c_width -- sigma in column direction
    """

    def check_if_unique(iterator):
        return len(set(iterator)) == 1

    num_dims = []

    for k, v in e_vec.items():
        assert isinstance(k, int), "Error! Keys need to be int"
        num_dims.append(len(v))

    assert check_if_unique(num_dims), "Error! Unequal number of dimensions"

    tmp = np.zeros((emax, num_dims[0]))

    for k, v in e_vec.items():
        tmp[k, :] = copy(v)
    pd = np.dot(tmp, tmp.T)

    return pd

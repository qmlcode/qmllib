from typing import List, Optional

import numpy as np
from numpy import int64, ndarray

from .fslatm import fget_sbop, fget_sbop_local, fget_sbot, fget_sbot_local


def update_m(obj, ia, rcut=9.0, pbc=None):
    """
    retrieve local structure around atom ia
    for periodic systems (or very large system)
    """

    raise NotImplementedError("Function is in-complete")

    # zs, coords, c = obj
    # v1, v2, v3 = c
    # vs = spatial_distance.norm(c, axis=0)

    # nns = []
    # for i, vi in enumerate(vs):
    #     # n1_doulbe = rcut / li  # what is li
    #     # n1 = int(n1_doulbe)
    #     # if n1 - n1_doulbe == 0:
    #     #     n1s = (
    #     #         range(-n1, n1 + 1)
    #     #         if pbc[i]
    #     #         else [
    #     #             0,
    #     #         ]
    #     #     )
    #     # elif n1 == 0:
    #     #     n1s = (
    #     #         [-1, 0, 1]
    #     #         if pbc[i]
    #     #         else [
    #     #             0,
    #     #         ]
    #     #     )
    #     # else:
    #     #     n1s = (
    #     #         range(-n1 - 1, n1 + 2)
    #     #         if pbc[i]
    #     #         else [
    #     #             0,
    #     #         ]
    #     #     )

    #     # nns.append(n1s)

    # n1s, n2s, n3s = nns

    # n123s_ = np.array(list(itertools.product(n1s, n2s, n3s)))
    # n123s = []
    # for n123 in n123s_:
    #     n123u = list(n123)
    #     if n123u != [0, 0, 0]:
    #         n123s.append(n123u)

    # nau = len(n123s)
    # n123s = np.array(n123s, np.float64)

    # na = len(zs)
    # cia = coords[ia]

    # if na == 1:
    #     ds = np.array([[0.0]])
    # else:
    #     ds = spatial_distance.squareform(spatial_distance.pdist(coords))

    # zs_u = []
    # coords_u = []
    # zs_u.append(zs[ia])
    # coords_u.append(coords[ia])
    # for i in range(na):
    #     di = ds[i, ia]
    #     if (di > 0) and (di <= rcut):
    #         zs_u.append(zs[i])
    #         coords_u.append(coords[ia])

    #         # add new coords by translation
    #         ts = np.zeros((nau, 3))
    #         for iau in range(nau):
    #             ts[iau] = np.dot(n123s[iau], c)

    #         coords_iu = coords[i] + ts  # np.dot(n123s, c)
    #         dsi = spatial_distance.norm(coords_iu - cia, axis=1)
    #         filt = np.logical_and(dsi > 0, dsi <= rcut)
    #         nx = filt.sum()
    #         zs_u += [
    #             zs[i],
    #         ] * nx
    #         coords_u += [
    #             list(coords_iu[filt, :]),
    #         ]

    # obj_u = [zs_u, coords_u]

    # return obj_u


def get_boa(z1: int64, zs_: ndarray) -> ndarray:
    return z1 * np.array(
        [
            (zs_ == z1).sum(),
        ]
    )
    # return -0.5*z1**2.4*np.array( [(zs_ == z1).sum(), ])


def get_sbop(
    mbtype: List[int64],
    obj: List[ndarray],
    iloc: bool = False,
    ia: Optional[int] = None,
    normalize: bool = True,
    sigma: float = 0.05,
    rcut: float = 4.8,
    dgrid: float = 0.03,
    pbc: str = "000",
    rpower: int = 6,
) -> ndarray:
    """
    two-body terms

    :param obj: molecule object, consisting of two parts: [ zs, coords ]
    :type obj: list
    """

    z1, z2 = mbtype
    zs, coords, c = obj

    if iloc and ia is None:
        raise ValueError("Please specify za and ia")

    if pbc != "000":
        raise NotImplementedError("Periodic boundary conditions not implemented")
        # if rcut < 9.0:
        #     raise ValueError()
        # assert iloc, "#ERROR: for periodic system, plz use atomic rpst"
        # zs, coords = update_m(obj, ia, rcut=rcut, pbc=pbc)

        # after update of `m, the query atom `ia will become the first atom
        # ia = 0

    # bop potential distribution
    r0 = 0.1
    nx = int((rcut - r0) / dgrid) + 1

    coeff = 1 / np.sqrt(2 * sigma**2 * np.pi) if normalize else 1.0

    if iloc:
        ys = fget_sbop_local(coords, zs, ia, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower)
    else:
        ys = fget_sbop(coords, zs, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower)

    return ys


def get_sbot(
    mbtype: List[int64],
    obj: List[ndarray],
    iloc: bool = False,
    ia: Optional[int] = None,
    normalize: bool = True,
    sigma: float = 0.05,
    rcut: float = 4.8,
    dgrid: float = 0.0262,
    pbc: str = "000",
) -> ndarray:
    """
    sigma -- standard deviation of gaussian distribution centered on a specific angle
            defaults to 0.05 (rad), approximately 3 degree
    dgrid    -- step of angle grid
            defaults to 0.0262 (rad), approximately 1.5 degree
    """

    z1, z2, z3 = mbtype
    zs, coords, c = obj

    if iloc and ia is None:
        raise ValueError("Please specify za and ia")

    if pbc != "000":
        raise NotImplementedError("Periodic boundary conditions not implemented")
        # assert iloc, "#ERROR: for periodic system, plz use atomic rpst"
        # zs, coords = update_m(obj, ia, rcut=rcut, pbc=pbc)

        # # after update of `m, the query atom `ia will become the first atom
        # ia = 0

    # for a normalized gaussian distribution, u should multiply this coeff
    coeff = 1 / np.sqrt(2 * sigma**2 * np.pi) if normalize else 1.0

    # Setup grid in Python
    d2r = np.pi / 180  # degree to rad
    a0 = -20.0 * d2r
    a1 = np.pi + 20.0 * d2r
    nx = int((a1 - a0) / dgrid) + 1

    if iloc:
        ys = fget_sbot_local(coords, zs, ia, z1, z2, z3, rcut, nx, dgrid, sigma, coeff)
    else:
        ys = fget_sbot(coords, zs, z1, z2, z3, rcut, nx, dgrid, sigma, coeff)

    return ys

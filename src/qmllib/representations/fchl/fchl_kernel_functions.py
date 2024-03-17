from __future__ import division, print_function

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.special import binom, factorial

from .ffchl_module import ffchl_kernel_types as kt


def get_gaussian_parameters(
    tags: Optional[Dict[str, List[float]]]
) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "sigma": [2.5],
        }

    parameters = np.array(tags["sigma"])

    for i in range(len(parameters)):
        parameters[i] = -0.5 / (parameters[i]) ** 2

    np.resize(parameters, (1, len(tags["sigma"])))

    n_kernels = len(tags["sigma"])

    return kt.gaussian, parameters, n_kernels


def get_linear_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "c": [0.0],
        }

    parameters = np.array(tags["c"])

    np.resize(parameters, (1, len(tags["c"])))

    n_kernels = len(tags["c"])

    return kt.linear, parameters, n_kernels


def get_polynomial_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {"alpha": [1.0], "c": [0.0], "d": [1.0]}

    parameters = np.array([tags["alpha"], tags["c"], tags["d"]]).T
    assert len(tags["alpha"]) == len(tags["c"])
    assert len(tags["alpha"]) == len(tags["d"])

    n_kernels = len(tags["alpha"])
    return kt.polynomial, parameters, n_kernels


def get_sigmoid_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "alpha": [1.0],
            "c": [0.0],
        }

    parameters = np.array(
        [
            tags["alpha"],
            tags["c"],
        ]
    ).T
    assert len(tags["alpha"]) == len(tags["c"])
    n_kernels = len(tags["alpha"])

    return kt.sigmoid, parameters, n_kernels


def get_multiquadratic_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "c": [0.0],
        }

    parameters = np.array(
        [
            tags["c"],
        ]
    ).T

    np.resize(parameters, (1, len(tags["c"])))
    n_kernels = len(tags["c"])

    return kt.multiquadratic, parameters, n_kernels


def get_inverse_multiquadratic_parameters(
    tags: Dict[str, List[float]]
) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "c": [0.0],
        }

    parameters = np.array(
        [
            tags["c"],
        ]
    ).T

    np.resize(parameters, (1, len(tags["c"])))
    n_kernels = len(tags["c"])

    return kt.inv_multiquadratic, parameters, n_kernels


def get_bessel_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {"sigma": [1.0], "v": [1.0], "n": [1.0]}

    parameters = np.array([tags["sigma"], tags["v"], tags["n"]]).T
    assert len(tags["sigma"]) == len(tags["v"])
    assert len(tags["sigma"]) == len(tags["n"])

    n_kernels = len(tags["sigma"])

    return kt.bessel, parameters, n_kernels


def get_l2_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "alpha": [1.0],
            "c": [0.0],
        }

    parameters = np.array(
        [
            tags["alpha"],
            tags["c"],
        ]
    ).T
    assert len(tags["alpha"]) == len(tags["c"])
    n_kernels = len(tags["alpha"])

    return kt.l2, parameters, n_kernels


def get_matern_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "sigma": [10.0],
            "n": [2.0],
        }

    assert len(tags["sigma"]) == len(tags["n"])
    n_kernels = len(tags["sigma"])

    n_max = int(max(tags["n"])) + 1

    parameters = np.zeros((2 + n_max, n_kernels))

    for i in range(n_kernels):
        parameters[0, i] = tags["sigma"][i]
        parameters[1, i] = tags["n"][i]

        n = int(tags["n"][i])
        for k in range(0, n + 1):
            parameters[2 + k, i] = float(factorial(n + k) * binom(n, k)) / factorial(2 * n)

    parameters = parameters.T

    return kt.matern, parameters, n_kernels


def get_cauchy_parameters(tags: Dict[str, List[float]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "sigma": [1.0],
        }

    parameters = np.array(
        [
            tags["sigma"],
        ]
    ).T

    np.resize(parameters, (1, len(tags["sigma"])))
    n_kernels = len(tags["sigma"])

    return kt.cauchy, parameters, n_kernels


def get_polynomial2_parameters(tags: Dict[str, List[List[float]]]) -> Tuple[ndarray, ndarray, int]:

    if tags is None:
        tags = {
            "coeff": [[1.0, 1.0, 1.0]],
        }

    parameters = np.zeros((10, len(tags["coeff"])))

    for i, c in enumerate(tags["coeff"]):
        for j, v in enumerate(c):
            parameters[j, i] = v

    n_kernels = len(tags["coeff"])
    parameters = parameters.T
    return kt.polynomial2, parameters, n_kernels


def get_kernel_parameters(
    name: str, tags: Optional[Union[Dict[str, List[List[float]]], Dict[str, List[float]]]]
) -> Tuple[ndarray, ndarray, int]:

    parameters = None
    idx = kt.gaussian
    n_kernels = 1

    if name == "gaussian":
        idx, parameters, n_kernels = get_gaussian_parameters(tags)

    elif name == "linear":
        idx, parameters, n_kernels = get_linear_parameters(tags)

    elif name == "polynomial":
        idx, parameters, n_kernels = get_polynomial_parameters(tags)

    elif name == "sigmoid":
        idx, parameters, n_kernels = get_sigmoid_parameters(tags)

    elif name == "multiquadratic":
        idx, parameters, n_kernels = get_multiquadratic_parameters(tags)

    elif name == "inverse-multiquadratic":
        idx, parameters, n_kernels = get_inverse_multiquadratic_parameters(tags)

    elif name == "bessel":
        idx, parameters, n_kernels = get_bessel_parameters(tags)

    elif name == "l2":
        idx, parameters, n_kernels = get_l2_parameters(tags)

    elif name == "matern":
        idx, parameters, n_kernels = get_matern_parameters(tags)

    elif name == "cauchy":
        idx, parameters, n_kernels = get_cauchy_parameters(tags)

    elif name == "polynomial2":
        idx, parameters, n_kernels = get_polynomial2_parameters(tags)

    else:

        print("QML ERROR: Unsupported kernel specification,", name)
        exit()

    return idx, parameters, n_kernels

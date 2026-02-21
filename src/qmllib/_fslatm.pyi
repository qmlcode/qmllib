import typing

import numpy
import numpy.typing

def fget_sbop(
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nuclear_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    z1: typing.SupportsInt,
    z2: typing.SupportsInt,
    rcut: typing.SupportsFloat,
    nx: typing.SupportsInt,
    dgrid: typing.SupportsFloat,
    sigma: typing.SupportsFloat,
    coeff: typing.SupportsFloat,
    rpower: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_sbop_local(
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nuclear_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    ia_python: typing.SupportsInt,
    z1: typing.SupportsInt,
    z2: typing.SupportsInt,
    rcut: typing.SupportsFloat,
    nx: typing.SupportsInt,
    dgrid: typing.SupportsFloat,
    sigma: typing.SupportsFloat,
    coeff: typing.SupportsFloat,
    rpower: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_sbot(
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nuclear_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    z1: typing.SupportsInt,
    z2: typing.SupportsInt,
    z3: typing.SupportsInt,
    rcut: typing.SupportsFloat,
    nx: typing.SupportsInt,
    dgrid: typing.SupportsFloat,
    sigma: typing.SupportsFloat,
    coeff: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_sbot_local(
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nuclear_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    ia_python: typing.SupportsInt,
    z1: typing.SupportsInt,
    z2: typing.SupportsInt,
    z3: typing.SupportsInt,
    rcut: typing.SupportsFloat,
    nx: typing.SupportsInt,
    dgrid: typing.SupportsFloat,
    sigma: typing.SupportsFloat,
    coeff: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...

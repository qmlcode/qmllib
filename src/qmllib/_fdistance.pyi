import typing

import numpy
import numpy.typing

def fl2_distance(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fmanhattan_distance(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fp_distance_double(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    p: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fp_distance_integer(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    p: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...

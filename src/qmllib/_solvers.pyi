import typing

import numpy
import numpy.typing

def fbkf_invert(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fbkf_solve(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> None: ...
def fcho_invert(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fcho_solve(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> None: ...
def fsvd_solve(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    la: typing.SupportsInt,
    rcond: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fqrlq_solve(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    y: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    la: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fcond(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> float: ...
def fcond_ge(
    A: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> float: ...

import typing

import numpy
import numpy.typing

def fatomic_local_gradient_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    na1: typing.SupportsInt,
    naq2: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fatomic_local_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    na1: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgaussian_process_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    na1: typing.SupportsInt,
    na2: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgdml_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    na1: typing.SupportsInt,
    na2: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fglobal_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def flocal_gradient_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    naq2: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def flocal_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def flocal_kernels(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    nm2: typing.SupportsInt,
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nsigmas: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fsymmetric_gaussian_process_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    na1: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fsymmetric_gdml_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    dx1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    na1: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fsymmetric_local_kernel(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fsymmetric_local_kernels(
    x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nm1: typing.SupportsInt,
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nsigmas: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...

import typing

import numpy
import numpy.typing

def fgaussian_kernel(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgaussian_kernel_symmetric(
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], sigma: typing.SupportsFloat
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_local_kernels_gaussian(
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_local_kernels_laplacian(
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_vector_kernels_gaussian(
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_vector_kernels_gaussian_symmetric(
    q: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    n: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_vector_kernels_laplacian(
    q1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    q2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    n1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    n2: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fget_vector_kernels_laplacian_symmetric(
    q: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    n: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    sigmas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fkpca(
    k: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    n: typing.SupportsInt,
    centering: bool,
) -> numpy.typing.NDArray[numpy.float64]: ...
def flaplacian_kernel(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    sigma: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def flaplacian_kernel_symmetric(
    x: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], sigma: typing.SupportsFloat
) -> numpy.typing.NDArray[numpy.float64]: ...
def flinear_kernel(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fmatern_kernel_l2(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    sigma: typing.SupportsFloat,
    order: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fsargan_kernel(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    sigma: typing.SupportsFloat,
    gammas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]: ...
def fwasserstein_kernel(
    a: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    na: typing.SupportsInt,
    b: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nb: typing.SupportsInt,
    sigma: typing.SupportsFloat,
    p: typing.SupportsInt,
    q: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...

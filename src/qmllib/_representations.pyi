import typing

import numpy
import numpy.typing

def fgenerate_atomic_coulomb_matrix(
    central_atom_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    central_natoms: typing.SupportsInt,
    atomic_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    natoms: typing.SupportsInt,
    nmax: typing.SupportsInt,
    cent_cutoff: typing.SupportsFloat,
    cent_decay: typing.SupportsFloat,
    int_cutoff: typing.SupportsFloat,
    int_decay: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgenerate_bob(
    atomic_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nuclear_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    id: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    nmax: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    ncm: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgenerate_coulomb_matrix(
    atomic_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nmax: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgenerate_eigenvalue_coulomb_matrix(
    atomic_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nmax: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgenerate_local_coulomb_matrix(
    central_atom_indices: typing.Annotated[numpy.typing.ArrayLike, numpy.int32],
    central_natoms: typing.SupportsInt,
    atomic_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    natoms: typing.SupportsInt,
    nmax: typing.SupportsInt,
    cent_cutoff: typing.SupportsFloat,
    cent_decay: typing.SupportsFloat,
    int_cutoff: typing.SupportsFloat,
    int_decay: typing.SupportsFloat,
) -> numpy.typing.NDArray[numpy.float64]: ...
def fgenerate_unsorted_coulomb_matrix(
    atomic_charges: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    coordinates: typing.Annotated[numpy.typing.ArrayLike, numpy.float64],
    nmax: typing.SupportsInt,
) -> numpy.typing.NDArray[numpy.float64]: ...

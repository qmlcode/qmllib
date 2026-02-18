# TODO: Convert these modules from f2py to pybind11
# from qmllib.representations.arad import generate_arad
from qmllib.representations.fchl import (
    generate_fchl18,
    generate_fchl18_displaced,
    generate_fchl18_displaced_5point,
    generate_fchl18_electric_field,
)
from qmllib.representations.representations import (
    generate_acsf,
    generate_bob,
    generate_coulomb_matrix,
    generate_coulomb_matrix_atomic,
    generate_coulomb_matrix_eigenvalue,
    generate_fchl19,
    generate_slatm,
    get_slatm_mbtypes,
)

__all__ = [
    "generate_fchl18",
    "generate_fchl18_displaced",
    "generate_fchl18_displaced_5point",
    "generate_fchl18_electric_field",
    "generate_acsf",
    "generate_bob",
    "generate_coulomb_matrix",
    "generate_coulomb_matrix_atomic",
    "generate_coulomb_matrix_eigenvalue",
    "generate_fchl19",
    "generate_slatm",
    "get_slatm_mbtypes",
]

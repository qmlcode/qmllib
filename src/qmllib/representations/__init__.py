# TODO: Convert these modules from f2py to pybind11
# from qmllib.representations.arad import generate_arad  # noqa:403
# from qmllib.representations.fchl import (  # noqa:F403
#     generate_fchl18,
#     generate_fchl18_displaced,
#     generate_fchl18_displaced_5point,
#     generate_fchl18_electric_field,
# )
from qmllib.representations.representations import (  # noqa:F403
    generate_acsf,
    generate_fchl19,
    generate_slatm,
    get_slatm_mbtypes,
    generate_bob,
    generate_coulomb_matrix,
    generate_coulomb_matrix_atomic,
    generate_coulomb_matrix_eigenvalue,
)

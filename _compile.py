""" Compile script for Fortran """

import os
import subprocess
import sys
from pathlib import Path

f90_modules = {
    "representations/frepresentations": ["frepresentations.f90"],
    "representations/facsf": ["facsf.f90"],
    "representations/fslatm": ["fslatm.f90"],
    "representations/arad/farad_kernels": ["farad_kernels.f90"],
    "representations/fchl/ffchl_module": [
        "ffchl_module.f90",
        "ffchl_scalar_kernels.f90",
        "ffchl_kernel_types.f90",
        "ffchl_kernels.f90",
        "ffchl_electric_field_kernels.f90",
        "ffchl_force_kernels.f90",
    ],
    "solvers/fsolvers": ["fsolvers.f90"],
    "kernels/fdistance": ["fdistance.f90"],
    "kernels/fkernels": ["fkernels.f90", "fkpca.f90"],
    "kernels/fgradient_kernels": ["fgradient_kernels.f90"],
}


def find_mkl():

    return


def find_flags(fcc: str):
    """Find compiler flags"""

    # TODO Find math lib
    # TODO Find os

    # -lgomp", "-lpthread", "-lm", "-ldl
    # ["-L${MKLROOT}/lib/intel64", "-lmkl_rt"]

    # COMPILER_FLAGS = ["-O3", "-fopenmp", "-m64", "-march=native", "-fPIC",
    #                 "-Wno-maybe-uninitialized", "-Wno-unused-function", "-Wno-cpp"]
    # LINKER_FLAGS = ["-lgomp"]

    extra_flags = ["-lgomp", "-lpthread", "-lm", "-ldl"]

    flags = ["-L/usr/lib/", "-lblas", "-llapack"] + extra_flags

    return flags


def find_fcc():
    """Find the fortran compiler. Either gnu or intel"""

    # fcc = "ifort"
    fcc = "gfortran"

    return fcc


def main():
    """Compile f90 in src/qmllib"""

    fcc = find_fcc()
    flags = find_flags(fcc)

    os.environ["FCC"] = fcc

    for module_name, module_sources in f90_modules.items():

        path = Path(module_name)
        parent = path.parent
        stem = path.stem

        cwd = Path("src/qmllib") / parent
        cmd = (
            [sys.executable, "-m", "numpy.f2py", "-c"] + flags + module_sources + ["-m", str(stem)]
        )
        print(cwd, " ".join(cmd))

        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        stdout = proc.stdout
        stderr = proc.stderr
        exitcode = proc.returncode

        if exitcode > 0:
            print(stdout)
            print()
            print(stderr)
            exit(exitcode)


if __name__ == "__main__":
    main()

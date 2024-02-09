""" Compile script for Fortran """

import os
import subprocess
from pathlib import Path

f90_modules = {
    "representations/frepresentations": ["frepresentations.f90"],
    "representations/facsf": ["facsf.f90"],
    "representations/fslatm": ["fslatm.f90"],
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

    flags = ["-L/usr/lib/", "-lblas", "-llapack"]

    return flags


def find_fcc():
    """Find the fortran compiler. Either gnu or intel"""

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
        cmd = "python -m numpy.f2py".split() + ["-c"] + flags + module_sources + ["-m", str(stem)]
        print(cwd, " ".join(cmd))

        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        log = proc.stdout
        error = proc.stderr
        exitcode = proc.returncode

        if exitcode > 0:
            print(log)
            print()
            print(error)
            exit(exitcode)

    exit(0)


if __name__ == "__main__":
    main()

""" Compile script for Fortran """

import os
import subprocess

f90_modules = {
    "representations.frepresentations": ["representations/frepresentations.f90"],
    "representations.facsf": ["representations/facsf.f90"],
    "representations.fslatm": ["representations/fslatm.f90"],
    # "": "representations/fchl/ffchl_module.f90",
    # "": "representations/fchl/ffchl_kernels.f90",
    # "": "representations/fchl/ffchl_electric_field_kernels.f90",
    # "": "representations/fchl/ffchl_kernel_types.f90",
    # "": "representations/fchl/ffchl_force_kernels.f90",
    # "": "representations/fchl/ffchl_scalar_kernels.f90",
    "solvers.fsolvers": ["solvers/fsolvers.f90"],
    "kernels.fdistance": ["kernels/fdistance.f90"],
    "kernels.fkernels": ["kernels/fkernels.f90", "kernels/fkpca.f90"],
    "kernels.fgradient_kernels": ["kernels/fgradient_kernels.f90"],
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

        cmd = f"python -m numpy.f2py {' '.join(flags)} -c {' '.join(module_sources)} -m {module_name}"
        print(cmd)

        proc = subprocess.run(cmd.split(), cwd="src/qmllib", capture_output=True, text=True)
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

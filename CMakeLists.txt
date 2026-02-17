cmake_minimum_required(VERSION 3.18)
project(qmllib LANGUAGES C CXX Fortran)

# Python + pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create a common interface target for kernels that need pybind11 headers
add_library(qmllib_common INTERFACE)
target_link_libraries(qmllib_common INTERFACE pybind11::headers Python::Module)

# Fortran solvers as an object library
add_library(qmllib_solvers OBJECT src/qmllib/solvers/fsolvers.f90)
set_property(TARGET qmllib_solvers PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran representations as an object library
add_library(qmllib_representations OBJECT src/qmllib/representations/frepresentations.f90)
set_property(TARGET qmllib_representations PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran utils as an object library
add_library(qmllib_utils OBJECT src/qmllib/utils/fsettings.f90)
set_property(TARGET qmllib_utils PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran kernels as an object library
add_library(qmllib_fkernels OBJECT 
    src/qmllib/kernels/fkpca.f90
    src/qmllib/kernels/fkwasserstein.f90
    src/qmllib/kernels/fkernels.f90
)
set_property(TARGET qmllib_fkernels PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran distance functions as an object library
add_library(qmllib_fdistance OBJECT src/qmllib/kernels/fdistance.f90)
set_property(TARGET qmllib_fdistance PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran gradient kernels as an object library
add_library(qmllib_fgradient_kernels OBJECT src/qmllib/kernels/fgradient_kernels.f90)
set_property(TARGET qmllib_fgradient_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran ACSF/FCHL representations as an object library
add_library(qmllib_facsf OBJECT src/qmllib/representations/facsf.f90)
set_property(TARGET qmllib_facsf PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran SLATM representations as an object library
add_library(qmllib_fslatm OBJECT src/qmllib/representations/fslatm.f90)
set_property(TARGET qmllib_fslatm PROPERTY POSITION_INDEPENDENT_CODE ON)

# Fortran FCHL representations as an object library
add_library(qmllib_ffchl OBJECT 
    src/qmllib/representations/fchl/ffchl_kernel_types.f90
    src/qmllib/representations/fchl/ffchl_kernels.f90
    src/qmllib/representations/fchl/ffchl_module.f90
    src/qmllib/representations/fchl/ffchl_scalar_kernels.f90
    src/qmllib/representations/fchl/ffchl_gradient_kernels.f90
    src/qmllib/representations/fchl/ffchl_hessian_kernels.f90
    src/qmllib/representations/fchl/ffchl_gaussian_process_kernels.f90
    src/qmllib/representations/fchl/ffchl_atomic_local_kernels.f90
    src/qmllib/representations/fchl/ffchl_force_alphas.f90
)
set_property(TARGET qmllib_ffchl PROPERTY POSITION_INDEPENDENT_CODE ON)

# Build the Python extension module for solvers
pybind11_add_module(_solvers MODULE
    src/qmllib/solvers/bindings_solvers.cpp
    $<TARGET_OBJECTS:qmllib_solvers>
)

set_target_properties(_solvers PROPERTIES OUTPUT_NAME "_solvers")

# Build the Python extension module for representations
pybind11_add_module(_representations MODULE
    src/qmllib/representations/bindings_representations.cpp
    $<TARGET_OBJECTS:qmllib_representations>
)

set_target_properties(_representations PROPERTIES OUTPUT_NAME "_representations")

# Build the Python extension module for utils
pybind11_add_module(_utils MODULE
    src/qmllib/utils/bindings_utils.cpp
    $<TARGET_OBJECTS:qmllib_utils>
)

set_target_properties(_utils PROPERTIES OUTPUT_NAME "_utils")

# Build the Python extension module for kernels (kpca and wasserstein)
pybind11_add_module(_fkernels MODULE
    src/qmllib/kernels/bindings_fkernels.cpp
    $<TARGET_OBJECTS:qmllib_fkernels>
)

set_target_properties(_fkernels PROPERTIES OUTPUT_NAME "_fkernels")

# Build the Python extension module for distance functions
pybind11_add_module(_fdistance MODULE
    src/qmllib/kernels/bindings_fdistance.cpp
    $<TARGET_OBJECTS:qmllib_fdistance>
)

set_target_properties(_fdistance PROPERTIES OUTPUT_NAME "_fdistance")

# Build the Python extension module for gradient kernels
pybind11_add_module(_fgradient_kernels MODULE
    src/qmllib/kernels/bindings_fgradient_kernels.cpp
    $<TARGET_OBJECTS:qmllib_fgradient_kernels>
)

set_target_properties(_fgradient_kernels PROPERTIES OUTPUT_NAME "_fgradient_kernels")

# Build the Python extension module for ACSF/FCHL representations
pybind11_add_module(_facsf MODULE
    src/qmllib/representations/bindings_facsf.cpp
    $<TARGET_OBJECTS:qmllib_facsf>
)

set_target_properties(_facsf PROPERTIES OUTPUT_NAME "_facsf")

# Build the Python extension module for SLATM representations
pybind11_add_module(_fslatm MODULE
    src/qmllib/representations/bindings_fslatm.cpp
    $<TARGET_OBJECTS:qmllib_fslatm>
)

set_target_properties(_fslatm PROPERTIES OUTPUT_NAME "_fslatm")

# Build the Python extension module for FCHL representations
pybind11_add_module(ffchl_module MODULE
    src/qmllib/representations/fchl/bindings_fchl_simple.cpp
    $<TARGET_OBJECTS:qmllib_ffchl>
)

set_target_properties(ffchl_module PROPERTIES OUTPUT_NAME "ffchl_module")

find_package(OpenMP)
if (OpenMP_Fortran_FOUND)
  target_link_libraries(_solvers PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_representations PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_utils PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_fkernels PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_fdistance PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_fgradient_kernels PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_facsf PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_fslatm PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(ffchl_module PRIVATE OpenMP::OpenMP_Fortran)
endif()

# Optional BLAS/LAPACK backends
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  target_link_libraries(_solvers PRIVATE ${ACCELERATE})
  target_link_libraries(_representations PRIVATE ${ACCELERATE})
  target_link_libraries(_fkernels PRIVATE ${ACCELERATE})
  target_link_libraries(ffchl_module PRIVATE ${ACCELERATE})
elseif(WIN32)
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(_solvers PRIVATE MKL::MKL)
  target_link_libraries(_representations PRIVATE MKL::MKL)
  target_link_libraries(_fkernels PRIVATE MKL::MKL)
  target_link_libraries(ffchl_module PRIVATE MKL::MKL)
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(_solvers PRIVATE BLAS::BLAS)
  target_link_libraries(_representations PRIVATE BLAS::BLAS)
  target_link_libraries(_fkernels PRIVATE BLAS::BLAS)
  target_link_libraries(ffchl_module PRIVATE BLAS::BLAS)
endif()

# Note: _fdistance doesn't need BLAS/LAPACK

# Compiler optimization flags (portable wheels)
if (CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  set(FORTRAN_OPT_FLAGS -O3 -ipo -xHost -fp-model fast=2 -no-prec-div -fno-alias -qopenmp)
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(FORTRAN_OPT_FLAGS -O3 -fopenmp -mcpu=native -mtune=native -ffast-math -ftree-vectorize)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CXX_OPT_FLAGS -O3 -march=native -ffast-math -fopenmp -mtune=native -ftree-vectorize)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CXX_OPT_FLAGS -O3 -qopt-report=3 -qopenmp -xHost)
endif()

# Apply optimization flags to Fortran object libraries
if(FORTRAN_OPT_FLAGS)
  target_compile_options(qmllib_solvers PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_representations PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_utils PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_fkernels PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_fdistance PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_fgradient_kernels PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_facsf PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_fslatm PRIVATE ${FORTRAN_OPT_FLAGS})
  target_compile_options(qmllib_ffchl PRIVATE ${FORTRAN_OPT_FLAGS})
endif()

# Apply optimization flags to C++ binding modules
if(CXX_OPT_FLAGS)
  target_compile_options(_solvers PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(_representations PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(_utils PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(_fkernels PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(_fdistance PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(_fgradient_kernels PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(_facsf PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(_fslatm PRIVATE ${CXX_OPT_FLAGS})
  target_compile_options(ffchl_module PRIVATE ${CXX_OPT_FLAGS})
endif()

# Install the compiled extension into the Python package and the Python shim
install(TARGETS _solvers _representations _utils _fkernels _fdistance _fgradient_kernels _facsf _fslatm
  LIBRARY DESTINATION qmllib   # Linux/macOS
  RUNTIME DESTINATION qmllib   # Windows (.pyd)
)

# Install FCHL module to the fchl subdirectory
install(TARGETS ffchl_module
  LIBRARY DESTINATION qmllib/representations/fchl   # Linux/macOS
  RUNTIME DESTINATION qmllib/representations/fchl   # Windows (.pyd)
)
install(DIRECTORY src/qmllib/ DESTINATION qmllib
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)


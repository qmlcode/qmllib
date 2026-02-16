cmake_minimum_required(VERSION 3.18)
project(qmllib LANGUAGES C CXX Fortran)

# Python + pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create a common interface target for kernels that need pybind11 headers
add_library(qmllib_common INTERFACE)
target_link_libraries(qmllib_common INTERFACE pybind11::headers Python::Module)

# Fortran kernels as an object library (for linking into the Python module)
add_library(qmllib_fortran OBJECT src/qmllib/kernels/kernel.f90)
set_property(TARGET qmllib_fortran PROPERTY POSITION_INDEPENDENT_CODE ON)

# Build the Python extension module via pybind11 and link the Fortran objects
pybind11_add_module(_qmllib MODULE
    src/qmllib/kernels/bindings.cpp
    $<TARGET_OBJECTS:qmllib_fortran>
)

# Ensure the built filename is exactly "_qmllib.*"
set_target_properties(_qmllib PROPERTIES OUTPUT_NAME "_qmllib")

# C++ kernel implementation (your new code)
add_library(qmllib_kernels OBJECT src/qmllib/kernels/kernels.cpp)
set_property(TARGET qmllib_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(qmllib_kernels PRIVATE qmllib_common)

# Build the Python extension module via pybind11 and link the Fortran objects
pybind11_add_module(_kernels MODULE
    src/qmllib/kernels/bindings_kernels.cpp
    $<TARGET_OBJECTS:qmllib_kernels>
)

set_target_properties(_kernels PROPERTIES OUTPUT_NAME "_kernels")

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()
if (OpenMP_Fortran_FOUND)
  target_link_libraries(_qmllib PRIVATE OpenMP::OpenMP_Fortran)
endif()

# Optional BLAS/LAPACK backends (enable later if needed)
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  target_link_libraries(_qmllib PRIVATE ${ACCELERATE})
  target_link_libraries(_kernels PRIVATE ${ACCELERATE})
elseif(WIN32)
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(_qmllib PRIVATE MKL::MKL)
  target_link_libraries(_kernels PRIVATE MKL::MKL)
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(_qmllib PRIVATE BLAS::BLAS)
  target_link_libraries(_kernels PRIVATE BLAS::BLAS)
endif()

# Conservative optimization flags (portable wheels). Override via env if you want.
if (CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  target_compile_options(qmllib_fortran PRIVATE -O3 -ipo -xHost -fp-model fast=2 -no-prec-div -fno-alias -qopenmp)
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(qmllib_fortran PRIVATE -O3 -fopenmp -mcpu=native -mtune=native -ffast-math -ftree-vectorize)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(qmllib_kernels PRIVATE -O3 -march=native -ffast-math -fopenmp -mtune=native -ftree-vectorize)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(qmllib_kernels PRIVATE -O3 -qopt-report=3 -qopenmp -xHost)
endif()

# Install the compiled extension into the Python package and the Python shim
install(TARGETS _qmllib _kernels
  LIBRARY DESTINATION qmllib   # Linux/macOS
  RUNTIME DESTINATION qmllib   # Windows (.pyd)
)
install(DIRECTORY src/qmllib/ DESTINATION qmllib
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)


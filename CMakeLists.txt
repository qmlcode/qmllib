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

find_package(OpenMP)
if (OpenMP_Fortran_FOUND)
  target_link_libraries(_solvers PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_representations PRIVATE OpenMP::OpenMP_Fortran)
  target_link_libraries(_utils PRIVATE OpenMP::OpenMP_Fortran)
endif()

# Optional BLAS/LAPACK backends
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  target_link_libraries(_solvers PRIVATE ${ACCELERATE})
  target_link_libraries(_representations PRIVATE ${ACCELERATE})
elseif(WIN32)
  find_package(MKL CONFIG REQUIRED)
  target_link_libraries(_solvers PRIVATE MKL::MKL)
  target_link_libraries(_representations PRIVATE MKL::MKL)
else()
  find_package(BLAS REQUIRED)
  target_link_libraries(_solvers PRIVATE BLAS::BLAS)
  target_link_libraries(_representations PRIVATE BLAS::BLAS)
endif()

# Install the compiled extension into the Python package and the Python shim
install(TARGETS _solvers _representations _utils
  LIBRARY DESTINATION qmllib   # Linux/macOS
  RUNTIME DESTINATION qmllib   # Windows (.pyd)
)
install(DIRECTORY src/qmllib/ DESTINATION qmllib
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)


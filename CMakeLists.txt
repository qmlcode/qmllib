cmake_minimum_required(VERSION 3.18)

# If building on Apple and user did not choose a compiler, default to Homebrew LLVM.
# This is so we don't need to export the path for clang explicitly before running cmake
if(APPLE)
  # Only set if not already set by -D... or environment/toolchain.
  if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(_HB_LLVM "/opt/homebrew/opt/llvm/bin")
    if(EXISTS "${_HB_LLVM}/clang" AND EXISTS "${_HB_LLVM}/clang++")
      set(CMAKE_C_COMPILER   "${_HB_LLVM}/clang"   CACHE FILEPATH "" FORCE)
      set(CMAKE_CXX_COMPILER "${_HB_LLVM}/clang++" CACHE FILEPATH "" FORCE)
    endif()
  endif()
endif()

project(qmllib LANGUAGES C CXX Fortran)

# ---- C++ standard -------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options -----------------------------------------------------------------
# Keep binaries portable by default; enable native tuning explicitly.
option(QMLLIB_USE_NATIVE "Enable -march/-mcpu=native tuning flags" OFF)

# Your current approach uses install_name_tool to add the Homebrew libomp rpath.
# Keep it as an option (default ON on macOS).
option(QMLLIB_USE_INSTALL_NAME_TOOL_RPATH "Use install_name_tool to add libomp rpath on macOS" ON)

# Prefer global PIC for all targets (object libs + modules).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Platform tweaks ----------------------------------------------------------
if(APPLE)
  # Required for "new lapack" in Accelerate
  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "" FORCE)
  add_compile_definitions(ACCELERATE_NEW_LAPACK)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)

  # Your existing libc++ / Homebrew LLVM runtime bits (kept as-is)
  # THis took me way too long to find out
  add_compile_options(-stdlib=libc++)
  add_link_options(
    -stdlib=libc++
    -L/opt/homebrew/opt/llvm/lib/c++
    -Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++
  )
endif()

# ---- Python + pybind11 --------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_library(qmllib_common INTERFACE)
target_link_libraries(qmllib_common INTERFACE pybind11::headers Python::Module)

# ---- OpenMP (required) --------------------------------------------------------
find_package(OpenMP REQUIRED)

# Helper to link OpenMP targets to something if found.
function(qmllib_link_openmp tgt)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  if(OpenMP_Fortran_FOUND)
    target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_Fortran)
  endif()
endfunction()

# ---- Optimization flags (portable by default) --------------------------------
function(qmllib_apply_cxx_opts tgt)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE
      -O3 -ffast-math -ftree-vectorize
      $<$<BOOL:${QMLLIB_USE_NATIVE}>:-march=native -mtune=native>
    )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    target_compile_options(${tgt} PRIVATE
      -O3
      $<$<BOOL:${QMLLIB_USE_NATIVE}>:-xHost>
    )
  endif()
endfunction()

function(qmllib_apply_fortran_opts tgt)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    target_compile_options(${tgt} PRIVATE
      -O3 -ipo -fp-model fast=2 -no-prec-div -fno-alias
      $<$<BOOL:${QMLLIB_USE_NATIVE}>:-xHost>
    )
  elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${tgt} PRIVATE
      -O3 -ffast-math -ftree-vectorize
      $<$<BOOL:${QMLLIB_USE_NATIVE}>:-mcpu=native -mtune=native>
    )
  endif()
endfunction()

# ---- macOS libomp runtime rpath hack (centralized) ----------------------------
function(qmllib_add_macos_libomp_rpath tgt)
  if(APPLE AND QMLLIB_USE_INSTALL_NAME_TOOL_RPATH)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND install_name_tool -add_rpath /opt/homebrew/opt/libomp/lib $<TARGET_FILE:${tgt}> || true
      VERBATIM
    )
  endif()
endfunction()

# ---- Fortran object libraries -------------------------------------------------
# Helper: create a Fortran object library and register it in a list.
set(QMLLIB_FORTRAN_OBJLIBS "")
function(qmllib_add_fortran_objlib name)
  add_library(${name} OBJECT ${ARGN})
  list(APPEND QMLLIB_FORTRAN_OBJLIBS ${name})
  set(QMLLIB_FORTRAN_OBJLIBS "${QMLLIB_FORTRAN_OBJLIBS}" PARENT_SCOPE)
endfunction()

qmllib_add_fortran_objlib(qmllib_solvers          src/qmllib/solvers/fsolvers.f90)
qmllib_add_fortran_objlib(qmllib_representations  src/qmllib/representations/frepresentations.f90)
qmllib_add_fortran_objlib(qmllib_utils            src/qmllib/utils/fsettings.f90)
qmllib_add_fortran_objlib(qmllib_fkernels
  src/qmllib/kernels/fkpca.f90
  src/qmllib/kernels/fkwasserstein.f90
  src/qmllib/kernels/fkernels.f90
)
qmllib_add_fortran_objlib(qmllib_fdistance        src/qmllib/kernels/fdistance.f90)
qmllib_add_fortran_objlib(qmllib_fgradient_kernels src/qmllib/kernels/fgradient_kernels.f90)
qmllib_add_fortran_objlib(qmllib_facsf            src/qmllib/representations/facsf.f90)
qmllib_add_fortran_objlib(qmllib_fslatm           src/qmllib/representations/fslatm.f90)
qmllib_add_fortran_objlib(qmllib_ffchl
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

# Apply Fortran opts + OpenMP to all Fortran object libs.
foreach(obj ${QMLLIB_FORTRAN_OBJLIBS})
  qmllib_apply_fortran_opts(${obj})
  qmllib_link_openmp(${obj})
endforeach()

# ---- Python extension modules -------------------------------------------------
set(QMLLIB_MODULES "")

function(qmllib_add_module modname binding_src fortran_objlib)
  # modname is the final target name (e.g. _solvers, ffchl_module)
  pybind11_add_module(${modname} MODULE
    ${binding_src}
    $<TARGET_OBJECTS:${fortran_objlib}>
  )
  target_link_libraries(${modname} PRIVATE qmllib_common)
  set_target_properties(${modname} PROPERTIES OUTPUT_NAME "${modname}")

  qmllib_apply_cxx_opts(${modname})
  qmllib_link_openmp(${modname})
  qmllib_add_macos_libomp_rpath(${modname})

  list(APPEND QMLLIB_MODULES ${modname})
  set(QMLLIB_MODULES "${QMLLIB_MODULES}" PARENT_SCOPE)
endfunction()

qmllib_add_module(_solvers            src/qmllib/solvers/bindings_solvers.cpp                     qmllib_solvers)
qmllib_add_module(_representations    src/qmllib/representations/bindings_representations.cpp    qmllib_representations)
qmllib_add_module(_utils              src/qmllib/utils/bindings_utils.cpp                         qmllib_utils)
qmllib_add_module(_fkernels           src/qmllib/kernels/bindings_fkernels.cpp                    qmllib_fkernels)
qmllib_add_module(_fdistance          src/qmllib/kernels/bindings_fdistance.cpp                   qmllib_fdistance)
qmllib_add_module(_fgradient_kernels  src/qmllib/kernels/bindings_fgradient_kernels.cpp          qmllib_fgradient_kernels)
qmllib_add_module(_facsf              src/qmllib/representations/bindings_facsf.cpp              qmllib_facsf)
qmllib_add_module(_fslatm             src/qmllib/representations/bindings_fslatm.cpp             qmllib_fslatm)

# Special case: module name doesn't start with underscore
qmllib_add_module(ffchl_module        src/qmllib/representations/fchl/bindings_fchl_simple.cpp   qmllib_ffchl)

# ---- BLAS/LAPACK backend selection -------------------------------------------
# Match your current behavior: Accelerate on macOS, MKL on Windows, BLAS elsewhere.
if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  set(QMLLIB_BLAS_TARGET "${ACCELERATE}")
elseif(WIN32)
  find_package(MKL CONFIG REQUIRED)
  set(QMLLIB_BLAS_TARGET MKL::MKL)
else()
  find_package(BLAS REQUIRED)
  set(QMLLIB_BLAS_TARGET BLAS::BLAS)
endif()

# Link BLAS to the subset that needs it (your current note: _fdistance doesn't).
set(QMLLIB_NEEDS_BLAS
  _solvers
  _representations
  _fkernels
  _fgradient_kernels
  ffchl_module
)

foreach(m ${QMLLIB_NEEDS_BLAS})
  target_link_libraries(${m} PRIVATE ${QMLLIB_BLAS_TARGET})
endforeach()

# ---- Install -----------------------------------------------------------------
install(TARGETS
  _solvers _representations _utils _fkernels _fdistance _fgradient_kernels _facsf _fslatm
  LIBRARY DESTINATION qmllib
  RUNTIME DESTINATION qmllib
)

install(TARGETS ffchl_module
  LIBRARY DESTINATION qmllib/representations/fchl
  RUNTIME DESTINATION qmllib/representations/fchl
)

install(DIRECTORY src/qmllib/ DESTINATION qmllib
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "*.pyi"
    PATTERN "py.typed"
  PATTERN "__pycache__" EXCLUDE
)

cmake_minimum_required(VERSION 3.10)
project(hacc_io_fork CXX)

get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(DEFAULT_LLVM_BUILD_ROOT "${REPO_ROOT}/llvm-infra/llvm-builds/apptainer-Debug")

set(LLVM_BUILD_ROOT "")
if(DEFINED ENV{LLVM_BUILD_DIR})
  set(LLVM_BUILD_ROOT "$ENV{LLVM_BUILD_DIR}")
elseif(LLVM_BUILD_DIR)
  set(LLVM_BUILD_ROOT "${LLVM_BUILD_DIR}")
else()
  set(LLVM_BUILD_ROOT "${DEFAULT_LLVM_BUILD_ROOT}")
endif()

set(LLVM_OPENMP_ROOT "${LLVM_BUILD_ROOT}")
if(DEFINED ENV{LLVM_INSTALL_DIR})
  set(LLVM_OPENMP_ROOT "$ENV{LLVM_INSTALL_DIR}")
elseif(DEFINED ENV{LLVM_ROOT})
  set(LLVM_OPENMP_ROOT "$ENV{LLVM_ROOT}")
endif()

# ---- baseline SPMD driver (POSIX / MPI-IO), upstream GLEAN class ----------
find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
  add_executable(testhacc_io testhacc_io.cc restartio_glean.cc)
  target_link_libraries(testhacc_io PUBLIC MPI::MPI_CXX)
else()
  message(WARNING "MPI CXX not found; skipping testhacc_io baseline target")
endif()

# ---- OMPFILE split-role variants (MPI-free; offload warmup) ----------------
# testhacc_ompfile: single-app-rank issuer (funnel posture)
# testhacc_target:  distributed issuers via target regions (fair posture)
# testior_target:   IOR -F geometry via target regions (fair posture)
set(OMPFILE_SEARCH_DIRS
  "${LLVM_BUILD_ROOT}/lib"
  "${LLVM_BUILD_ROOT}/lib/x86_64-unknown-linux-gnu"
  "${LLVM_BUILD_ROOT}/runtimes/runtimes-bins/openmp/libompfile"
  "${LLVM_OPENMP_ROOT}/lib"
  "${LLVM_OPENMP_ROOT}/lib/x86_64-unknown-linux-gnu"
  "${LLVM_OPENMP_ROOT}/runtimes/runtimes-bins/openmp/libompfile")
list(REMOVE_DUPLICATES OMPFILE_SEARCH_DIRS)
find_library(OMPFILE_LIBRARY NAMES ompfile PATHS ${OMPFILE_SEARCH_DIRS})
if(NOT OMPFILE_LIBRARY)
  message(FATAL_ERROR "Could not find libompfile.so in: ${OMPFILE_SEARCH_DIRS}")
endif()

foreach(_ompfile_bin IN ITEMS testhacc_ompfile testhacc_target testior_target)
  add_executable(${_ompfile_bin} ${_ompfile_bin}.cc)
  target_include_directories(${_ompfile_bin} PRIVATE
    "${REPO_ROOT}/llvm-infra/llvm-project/openmp/libompfile/include")
  target_compile_options(${_ompfile_bin} PRIVATE
    -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu)
  target_link_options(${_ompfile_bin} PRIVATE
    -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu)
  target_link_libraries(${_ompfile_bin} PRIVATE "${OMPFILE_LIBRARY}")
  foreach(_dir IN LISTS OMPFILE_SEARCH_DIRS)
    if(EXISTS "${_dir}")
      target_link_libraries(${_ompfile_bin} PRIVATE "-Wl,-rpath,${_dir}")
    endif()
  endforeach()
endforeach()

# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Findhip
# --------
#
# Find the HIP library
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the ``IMPORTED`` target ``hip::host``,
# if HIP has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``HIP_PLATFORM``
#   the target platform (e.g., amd or nvidia)
# ``HIP_COMPILER``
#   the target platform compiler (e.g., clang or nvcc)
# ``HIP_RUNTIME``
#   the target platform runtime (e.g, rocclr or cuda)
# ``HIP_VERSION``
#   the HIP library version
# ``hip_INCLUDE_DIRS``
#   include directories for HIP
# ``hip_FOUND``
#   true if HIP has been found and can be used

include(FindPackageHandleStandardArgs)

# Search for HIP installation
if(NOT hip_DIR)
    # Search in user specified path first
    find_path(
        hip_DIR
        NAMES bin/hipconfig
        PATHS
            ENV ROCM_PATH
            "$ENV{ROCM_PATH}/hip"
            ENV HIP_PATH
        NO_DEFAULT_PATH
        DOC "HIP installed location"
        )
    # Now search in default paths
    find_path(
        hip_DIR
        NAMES bin/hipconfig
        DOC "HIP installed location"
        )
    if(NOT EXISTS ${hip_DIR})
        if(hip_FIND_REQUIRED)
            message(FATAL_ERROR "Specify hip_DIR")
        elseif(NOT hip_FIND_QUIETLY)
            message("hip_DIR not found or specified")
        endif()
    endif()
    # And push it back to the cache
    set(hip_DIR ${hip_DIR} CACHE PATH "HIP installed location" FORCE)
endif()

# Find HIPCC executable
find_program(
    HIP_HIPCC_EXECUTABLE
    NAMES hipcc
    PATHS
        "${hip_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
    )
# Now search in default paths
find_program(HIP_HIPCC_EXECUTABLE hipcc)

# Find HIPCONFIG executable
find_program(
    HIP_HIPCONFIG_EXECUTABLE
    NAMES hipconfig
    PATHS
        "${hip_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH
    )
# Now search in default paths
find_program(HIP_HIPCONFIG_EXECUTABLE hipconfig)
if(NOT UNIX)
    set(HIP_HIPCONFIG_EXECUTABLE "${HIP_HIPCONFIG_EXECUTABLE}.bat")
    set(HIP_HIPCC_EXECUTABLE "${HIP_HIPCC_EXECUTABLE}.bat")
endif()
mark_as_advanced(HIP_HIPCONFIG_EXECUTABLE)
mark_as_advanced(HIP_HIPCC_EXECUTABLE)

if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_VERSION)
    # Compute the version
    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --version
        OUTPUT_VARIABLE _hip_version
        ERROR_VARIABLE _hip_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        )
    if(NOT _hip_error)
        set(HIP_VERSION ${_hip_version} CACHE STRING "Version of HIP")
    else()
        set(HIP_VERSION "0.0.0" CACHE STRING "Version of HIP")
    endif()
    mark_as_advanced(HIP_VERSION)
endif()
if(HIP_VERSION)
    string(REPLACE "." ";" _hip_version_list "${HIP_VERSION}")
    list(GET _hip_version_list 0 HIP_VERSION_MAJOR)
    list(GET _hip_version_list 1 HIP_VERSION_MINOR)
    list(GET _hip_version_list 2 HIP_VERSION_PATCH)
    set(HIP_VERSION_STRING "${HIP_VERSION}")
endif()

if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_PLATFORM)
    # Compute the platform
    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --platform
        OUTPUT_VARIABLE _hip_platform
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    set(HIP_PLATFORM ${_hip_platform} CACHE STRING "HIP platform")
    mark_as_advanced(HIP_PLATFORM)
endif()

if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_COMPILER)
    # Compute the compiler
    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --compiler
        OUTPUT_VARIABLE _hip_compiler
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    set(HIP_COMPILER ${_hip_compiler} CACHE STRING "HIP compiler")
    mark_as_advanced(HIP_COMPILER)
endif()

if(HIP_HIPCONFIG_EXECUTABLE AND NOT HIP_RUNTIME)
    # Compute the runtime
    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --runtime
        OUTPUT_VARIABLE _hip_runtime
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    set(HIP_RUNTIME ${_hip_runtime} CACHE STRING "HIP runtime")
    mark_as_advanced(HIP_RUNTIME)
endif()

if(HIP_PLATFORM STREQUAL "amd")
    find_package(hip QUIET CONFIG PATHS ${HIP_PATH} ${ROCM_PATH} /opt/rocm)
    find_package_handle_standard_args(hip CONFIG_MODE)
else()
    find_path(hip_INCLUDE_DIR hip/hip_runtime.h)
    mark_as_advanced(hip_INCLUDE_DIR)
    set(hip_INCLUDE_DIRS "${hip_INCLUDE_DIR}")
    set(HIP_INCLUDE_DIRS "${hip_INCLUDE_DIR}")

    find_package_handle_standard_args(hip
        REQUIRED_VARS
            hip_DIR
            hip_INCLUDE_DIR
            HIP_HIPCC_EXECUTABLE
            HIP_HIPCONFIG_EXECUTABLE
        VERSION_VAR
            HIP_VERSION
        )

    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    find_package(CUDA REQUIRED)

    if(NOT TARGET hip::host)
        add_library(hip::host INTERFACE IMPORTED)
        set_target_properties(hip::host PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_NVCC__=1;__HIP_PLATFORM_NVIDIA__=1"
            INTERFACE_INCLUDE_DIRECTORIES "${hip_INCLUDE_DIRS};${CUDA_INCLUDE_DIRS}"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${hip_INCLUDE_DIRS};${CUDA_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${CUDA_LIBRARIES};Threads::Threads"
        )
    endif()
endif()

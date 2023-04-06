# ########################################################################
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

add_executable( hipsolver-bench client.cpp )

# Internal header includes
target_include_directories( hipsolver-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

# External header includes included as system files
target_include_directories( hipsolver-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

target_link_libraries( hipsolver-bench PRIVATE
  cblas
  lapack
  blas
  hipsolver-common
  clients-common
  roc::hipsolver
)
if( UNIX )
  target_link_libraries( hipsolver-bench PRIVATE hipsolver_fortran_client )
endif( )

add_armor_flags( hipsolver-bench "${ARMOR_LEVEL}" )

# need mf16c flag for float->half convertion
target_compile_options( hipsolver-bench PRIVATE -mf16c)

if( NOT USE_CUDA )
  target_link_libraries( hipsolver-bench PRIVATE hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipsolver-bench PRIVATE hip::${CUSTOM_TARGET} )
  endif( )

  if( UNIX AND CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries( hipsolver-bench PRIVATE -lpthread -lm )
  endif()
else( )
  target_compile_definitions( hipsolver-bench PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( hipsolver-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipsolver-bench PRIVATE ${CUDA_LIBRARIES} Threads::Threads )
endif( )

set_target_properties( hipsolver-bench PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( hipsolver-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
#add_dependencies( hipsolver-bench hipsolver-bench-common )

rocm_install(TARGETS hipsolver-bench COMPONENT benchmarks)

target_compile_definitions( hipsolver-bench PRIVATE HIPSOLVER_BENCH ROCM_USE_FLOAT16 )

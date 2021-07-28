# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

# Linking lapack library requires fortran flags
find_package( cblas REQUIRED CONFIG )

if( NOT TARGET hipsolver )
  find_package( hipsolver REQUIRED CONFIG PATHS /opt/rocm/hipsolver )
endif( )

set(hipsolver_benchmark_common
  ../common/lapack_host_reference.cpp
  ../common/hipsolver_datatype2string.cpp
  ../common/utility.cpp
)

add_executable( hipsolver-bench client.cpp ${hipsolver_benchmark_common} )

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

target_link_libraries( hipsolver-bench PRIVATE cblas lapack blas roc::hipsolver )
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

  if( UNIX AND CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
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

target_compile_definitions( hipsolver-bench PRIVATE HIPSOLVER_BENCH ROCM_USE_FLOAT16 )

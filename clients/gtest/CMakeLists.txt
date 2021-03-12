# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.64.0 1.64 )
set( Boost_USE_STATIC_LIBS OFF )
find_package( Boost COMPONENTS program_options )

if( NOT Boost_FOUND )
  message( STATUS "Dynamic boost libraries not found. Attempting to find static libraries " )
  set( Boost_USE_STATIC_LIBS ON )
  find_package( Boost COMPONENTS program_options )

  if( NOT Boost_FOUND )
    message( FATAL_ERROR "boost is a required dependency and is not found;  try adding boost path to CMAKE_PREFIX_PATH" )
  endif( )
endif( )

# Linking lapack library requires fortran flags
enable_language( Fortran )
find_package( cblas CONFIG REQUIRED )
if( NOT cblas_FOUND )
  message( FATAL_ERROR "cblas is a required dependency and is not found;  try adding cblas path to CMAKE_PREFIX_PATH" )
endif( )

if( NOT TARGET hipsolver )
  find_package( hipsolver CONFIG PATHS /opt/rocm/hipsolver )

  if( NOT hipsolver_FOUND )
    message( FATAL_ERROR "hipSOLVER is a required dependency and is not found; try adding hipsolver path to CMAKE_PREFIX_PATH")
  endif( )
endif( )

find_package( GTest REQUIRED )
include_directories(${GTEST_INCLUDE_DIRS})

set(hipsolver_test_source
  hipsolver_gtest_main.cpp
  getrf_gtest.cpp
)

set( hipsolver_test_common
  ../common/lapack_host_reference.cpp
  ../common/hipsolver_datatype2string.cpp
  ../common/utility.cpp
)

add_executable( hipsolver-test ${hipsolver_f90_source} ${hipsolver_test_source} ${hipsolver_test_common} )

target_include_directories( hipsolver-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )
target_link_libraries( hipsolver-test PRIVATE Threads::Threads )

target_compile_definitions( hipsolver-test PRIVATE GOOGLE_TEST )

# External header includes included as SYSTEM files
target_include_directories( hipsolver-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    ${ROCM_PATH}/hsa/include
)

target_link_libraries( hipsolver-test PRIVATE roc::hipsolver cblas lapack ${GTEST_LIBRARIES} ${Boost_LIBRARIES} hipsolver_fortran_client )

# need mf16c flag for float->half convertion
target_compile_options( hipsolver-test PRIVATE -mf16c )

if( NOT CUDA_FOUND )
  target_compile_definitions( hipsolver-test PRIVATE __HIP_PLATFORM_HCC__ )


  if( CUSTOM_TARGET )
    target_link_libraries( hipsolver-test PRIVATE hip::${CUSTOM_TARGET} hip::host )
  else( )
      target_link_libraries( hipsolver-test PRIVATE hip::host )
  endif( )

  if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$|.*/hipcc$" )
    # hip-clang needs specific flag to turn on pthread and m
    target_link_libraries( hipsolver-test PRIVATE -lpthread -lm )
  endif()
else( )
  target_compile_definitions( hipsolver-test PRIVATE __HIP_PLATFORM_NVCC__ )

  target_include_directories( hipsolver-test
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )

  target_link_libraries( hipsolver-test PRIVATE ${CUDA_LIBRARIES} Threads::Threads )
endif( )

set_target_properties( hipsolver-test PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( hipsolver-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# ########################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc.
# ########################################################################

# Linking lapack library requires fortran flags
find_package( cblas REQUIRED CONFIG )

if( NOT TARGET hipsolver )
  find_package( hipsolver REQUIRED CONFIG PATHS /opt/rocm/hipsolver )
endif( )

find_package( GTest REQUIRED )

set(hipsolver_test_source
  hipsolver_gtest_main.cpp
  getrs_gtest.cpp
  getrf_gtest.cpp
  gebrd_gtest.cpp
  gels_gtest.cpp
  geqrf_gtest.cpp
  gesv_gtest.cpp
  gesvd_gtest.cpp
  potrf_gtest.cpp
  potri_gtest.cpp
  potrs_gtest.cpp
  syevd_heevd_gtest.cpp
  syevj_heevj_gtest.cpp
  sygvd_hegvd_gtest.cpp
  sytrd_hetrd_gtest.cpp
  sytrf_gtest.cpp
  orgbr_ungbr_gtest.cpp
  orgqr_ungqr_gtest.cpp
  orgtr_ungtr_gtest.cpp
  ormqr_unmqr_gtest.cpp
  ormtr_unmtr_gtest.cpp
)

set( hipsolver_test_common
  ../common/lapack_host_reference.cpp
  ../common/hipsolver_datatype2string.cpp
  ../common/utility.cpp
)

add_executable( hipsolver-test ${hipsolver_test_source} ${hipsolver_test_common} )
if( UNIX )
  target_sources( hipsolver-test PRIVATE ${hipsolver_f90_source} )
endif( )

target_include_directories( hipsolver-test
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

add_armor_flags( hipsolver-test "${ARMOR_LEVEL}" )

target_compile_definitions( hipsolver-test PRIVATE GOOGLE_TEST )

# External header includes included as SYSTEM files
target_include_directories( hipsolver-test
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    ${ROCM_PATH}/hsa/include
)

target_link_libraries( hipsolver-test PRIVATE cblas lapack blas GTest::GTest Threads::Threads roc::hipsolver )
if( UNIX )
  target_link_libraries( hipsolver-test PRIVATE hipsolver_fortran_client )
endif( )

target_link_libraries(hipsolver-test PRIVATE
  $<BUILD_INTERFACE:hipsolver-common>
)

# need mf16c flag for float->half convertion
target_compile_options( hipsolver-test PRIVATE -mf16c )

if( NOT USE_CUDA )
  target_link_libraries( hipsolver-test PRIVATE hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipsolver-test PRIVATE hip::${CUSTOM_TARGET} )
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

if(WIN32)
  file(GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${cblas_DIR}/bin/*.dll
    ${GTest_DIR}/bin/*.dll
    $ENV{rocblas_DIR}/bin/*.dll
    $ENV{rocsolver_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/*.dll
    $ENV{HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
  )
  foreach(file_i ${third_party_dlls})
    add_custom_command(TARGET hipsolver-test
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/
    )
  endforeach()
  add_custom_command(TARGET hipsolver-test
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory $ENV{rocblas_DIR}/bin/library ${PROJECT_BINARY_DIR}/staging/library
  )
endif()

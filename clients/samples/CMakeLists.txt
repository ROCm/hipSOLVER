# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

add_executable( example-c-basic example_basic.c )
add_executable( example-cpp-basic example_basic.cpp )

# We test for C99 compatibility in the example-c.c test
set_source_files_properties(example_basic.c PROPERTIES LANGUAGE CXX)
set_source_files_properties(example_basic.c PROPERTIES COMPILE_FLAGS "-xc -std=c99")

# We test for C++11 compatibility in one of the samples
set_source_files_properties(example_basic.cpp PROPERTIES COMPILE_FLAGS "-std=c++11")

if( NOT TARGET hipsolver )
  find_package( hipsolver CONFIG PATHS ${ROCM_PATH}/hipsolver )
  if( NOT hipsolver_FOUND )
    message( FATAL_ERROR "hipSOLVER is a required dependency and is not found; try adding hipSOLVER path to CMAKE_PREFIX_PATH")
  endif( )
endif( )

foreach( exe example-c-basic;example-cpp-basic; )

  # External header includes included as SYSTEM files
  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  )

  target_include_directories( ${exe} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include> )

  target_compile_options( ${exe} PRIVATE -mf16c)

  target_link_libraries( ${exe} PRIVATE roc::hipsolver )

  if( NOT USE_CUDA )
    target_link_libraries( ${exe} PRIVATE hip::host )

    if( CUSTOM_TARGET )
      target_link_libraries( ${exe} PRIVATE hip::${CUSTOM_TARGET} )
    endif( )

  else( )
    target_compile_definitions( ${exe} PRIVATE __HIP_PLATFORM_NVCC__ )

    target_include_directories( ${exe}
      PRIVATE
        $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
    )

    target_link_libraries( ${exe} PRIVATE ${CUDA_LIBRARIES} )
  endif( )

  set_target_properties(${exe} PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties( ${exe} PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
  set_target_properties( ${exe} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

endforeach( )

# ########################################################################
# Copyright 2016-2022 Advanced Micro Devices, Inc.
# ########################################################################

# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

# ########################################################################
# Main
# ########################################################################
prepend_path( ".." hipsolver_headers_public relative_hipsolver_headers_public )

if( NOT USE_CUDA )
  set( hipsolver_source
    "${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipsolver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipsolver_compat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/hipsolver_compat_common.cpp"
  )
else( )
  set( hipsolver_source
    "${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipsolver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipsolver_compat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/hipsolver_compat_common.cpp"
  )
endif( )

set (hipsolver_f90_source
  hipsolver_module.f90
)

if( UNIX )
  # Create hipSOLVER Fortran module
  add_library(hipsolver_fortran ${hipsolver_f90_source})
endif( )

add_library( hipsolver
  ${hipsolver_source}
  ${relative_hipsolver_headers_public}
)
add_library( roc::hipsolver ALIAS hipsolver )

target_link_libraries(hipsolver PRIVATE
  $<BUILD_INTERFACE:hipsolver-common> # https://gitlab.kitware.com/cmake/cmake/-/issues/15415
)

add_armor_flags( hipsolver "${ARMOR_LEVEL}" )

if( WIN32 )
  if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS )
    add_custom_command( TARGET hipsolver
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipsolver>" "${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipsolver>"
    )
    if( ${CMAKE_BUILD_TYPE} MATCHES "Debug" )
      add_custom_command( TARGET hipsolver
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_BINARY_DIR}/staging/hipsolver-d.pdb" "${PROJECT_BINARY_DIR}/clients/staging/hipsolver-d.pdb"
      )
    endif( )
  endif( )
endif( )

# Build hipsolver from source on AMD platform
if( NOT USE_CUDA )
  if( NOT TARGET rocblas )
    if( CUSTOM_ROCBLAS )
      set ( ENV{rocblas_DIR} ${CUSTOM_ROCBLAS})
      find_package( rocblas REQUIRED CONFIG NO_CMAKE_PATH )
    else( )
      find_package( rocblas REQUIRED CONFIG PATHS /opt/rocm /opt/rocm/rocblas )
    endif( )
  endif( )

  if( NOT TARGET rocsolver )
    if( CUSTOM_ROCSOLVER )
      set ( ENV{rocsolver_DIR} ${CUSTOM_ROCSOLVER})
      find_package( rocsolver REQUIRED CONFIG NO_CMAKE_PATH )
    else( )
      find_package( rocsolver REQUIRED CONFIG PATHS /opt/rocm /opt/rocm/rocsolver /usr/local/rocsolver )
    endif( )
  endif( )

  target_link_libraries( hipsolver PRIVATE roc::rocblas roc::rocsolver hip::host )

  if( CUSTOM_TARGET )
    target_link_libraries( hipsolver PRIVATE hip::${CUSTOM_TARGET} )
  endif( )

else( )
  target_compile_definitions( hipsolver PRIVATE __HIP_PLATFORM_NVCC__ )

  target_link_libraries( hipsolver PRIVATE ${CUDA_cusolver_LIBRARY} )

  # External header includes included as system files
  target_include_directories( hipsolver
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )
endif( )

# Internal header includes
target_include_directories( hipsolver
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
          $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include/internal>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipsolver>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipsolver/internal>
          $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${CMAKE_CURRENT_SOURCE_DIR}
)

rocm_set_soversion( hipsolver ${hipsolver_SOVERSION} )
set_target_properties( hipsolver PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipsolver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( hipsolver PROPERTIES DEBUG_POSTFIX "-d" )

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
set_target_properties( hipsolver PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
generate_export_header( hipsolver EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipsolver/internal/hipsolver-export.h )

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/library/include ${PROJECT_BINARY_DIR}/include/hipsolver)
if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_file(
    internal/hipsolver-version.h internal/hipsolver-export.h
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include hipsolver/include
    ORIGINAL_FILES ${PROJECT_BINARY_DIR}/include/hipsolver/internal/hipsolver-version.h
  )
endif( )


# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( hipsolver PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS hipsolver
  INCLUDE
    ${CMAKE_BINARY_DIR}/include
)

if ( NOT USE_CUDA )
  rocm_export_targets(
    TARGETS roc::hipsolver
    DEPENDS PACKAGE hip
    NAMESPACE roc::
  )
else( )
  rocm_export_targets(
    TARGETS roc::hipsolver
    DEPENDS PACKAGE HIP
    NAMESPACE roc::
  )
endif( )

if( UNIX )
  # Force installation of .f90 module files
  install(FILES "hipsolver_module.f90"
          DESTINATION "include/hipsolver")
endif( )

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/hipsolver"
        DESTINATION "." )

  if ( NOT WIN32 )

    #Create SymLink for Fortran Object Module for backward compatibility
    rocm_install(
      CODE "
        set(PREFIX \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX})
	set(INPUT_FILE \${PREFIX}/include/hipsolver/hipsolver_module.f90)
        set(SYMLINK_LOCATIONS \${PREFIX}/hipsolver/include \${PREFIX}/include)
        foreach(LOCATION IN LISTS SYMLINK_LOCATIONS)
          file(MAKE_DIRECTORY \${LOCATION})
          execute_process(COMMAND ln -sfr \${INPUT_FILE} \${LOCATION})
          message(STATUS \"Created symlink in \${LOCATION} to \${INPUT_FILE}.\")
        endforeach()
        "
    )
  endif() #NOT WIN32
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()


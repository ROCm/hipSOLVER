# ########################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc.
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
  set( hipsolver_source "${CMAKE_CURRENT_SOURCE_DIR}/hcc_detail/hipsolver.cpp" )
else( )
  set( hipsolver_source "${CMAKE_CURRENT_SOURCE_DIR}/nvcc_detail/hipsolver.cpp" )
endif( )

set (hipsolver_f90_source
  hipsolver_module.f90
)

# Create hipSOLVER Fortran module
add_library(hipsolver_fortran ${hipsolver_f90_source})

add_library( hipsolver
  ${hipsolver_source}
  ${relative_hipsolver_headers_public}
)
add_library( roc::hipsolver ALIAS hipsolver )

add_armor_flags( hipsolver "${ARMOR_LEVEL}" )

# External header includes included as system files
target_include_directories( hipsolver
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

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
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
  )
endif( )

# Internal header includes
target_include_directories( hipsolver
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
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
generate_export_header( hipsolver EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipsolver-export.h )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( hipsolver PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS hipsolver
  INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
  PREFIX hipsolver
)
#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

if ( NOT USE_CUDA )
  rocm_export_targets(
    TARGETS roc::hipsolver
    PREFIX hipsolver
    DEPENDS PACKAGE hip
    NAMESPACE roc::
  )
else( )
  rocm_export_targets(
    TARGETS roc::hipsolver
    PREFIX hipsolver
    DEPENDS PACKAGE HIP
    NAMESPACE roc::
  )
endif( )

# Force installation of .f90 module files
install(FILES "hipsolver_module.f90"
        DESTINATION "hipsolver/include")

rocm_install_symlink_subdir( hipsolver )

macro (do_set_imagedrawing_source_dir)
  set(DO_ImageDrawing_SOURCE_DIR ${DO_SOURCE_DIR}/ImageDrawing)
endmacro (do_set_imagedrawing_source_dir)

macro (do_list_imagedrawing_source_files)
  # Master header file
  set(DO_ImageDrawing_MASTER_HEADER ${DO_SOURCE_DIR}/ImageDrawing.hpp)
  source_group("Master Header File" FILES ${DO_ImageDrawing_MASTER_HEADER})
  # Header files
  file(GLOB DO_ImageDrawing_HEADER_FILES
       ${DO_ImageDrawing_SOURCE_DIR}/*.hpp)
  # Source files
  file(GLOB DO_ImageDrawing_SOURCE_FILES
       ${DO_ImageDrawing_SOURCE_DIR}/*.cpp)
  # All header files here
  set(DO_ImageDrawing_HEADER_FILES
      ${DO_ImageDrawing_MASTER_HEADER}
      ${DO_ImageDrawing_HEADER_FILES})
endmacro (do_list_imagedrawing_source_files)

macro (do_load_packages_for_imagedrawing_library)
  include(${DO_Core_USE_FILE})
  include_directories(
    ${antigrain_DIR}/include
    ${jpeg_DIR}
    ${png_DIR}
    ${tiff_DIR})
endmacro (do_load_packages_for_imagedrawing_library)

macro (do_create_variables_for_imagedrawing_library)
  set(DO_ImageDrawing_LIBRARIES DO_ImageDrawing)
  set(DO_ImageDrawing_LINK_LIBRARIES "antigrain;jpeg;png;tiff")
endmacro (do_create_variables_for_imagedrawing_library)

do_load_packages_for_imagedrawing_library()

if (DO_USE_FROM_SOURCE)
  get_property(DO_ImageDrawing_ADDED GLOBAL PROPERTY _DO_ImageDrawing_INCLUDED)
  if (NOT DO_ImageDrawing_ADDED)
    do_set_imagedrawing_source_dir()
    do_list_imagedrawing_source_files()
    do_create_variables_for_imagedrawing_library()
    
    # Static library
    do_append_library(
      ImageDrawing STATIC
      "${DO_SOURCE_DIR}"
      "${DO_ImageDrawing_HEADER_FILES}"
      "${DO_ImageDrawing_SOURCE_FILES}"
      "${DO_ImageDrawing_LINK_LIBRARIES}"
    )
    do_set_specific_target_properties(DO_ImageDrawing DO_STATIC)

    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        ImageDrawing_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_ImageDrawing_HEADER_FILES}"
        "${DO_ImageDrawing_SOURCE_FILES}"
        "${DO_ImageDrawing_LINK_LIBRARIES}"
      )
      do_set_specific_target_properties(DO_ImageDrawing_SHARED DO_EXPORTS)
    endif ()
  endif ()
endif ()

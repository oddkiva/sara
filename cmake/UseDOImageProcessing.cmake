macro (do_set_imageprocessing_source_dir)
  set(DO_ImageProcessing_SOURCE_DIR ${DO_SOURCE_DIR}/ImageProcessing)
endmacro (do_set_imageprocessing_source_dir)

macro (do_list_imageprocessing_source_files)
  # Master header file
  set(DO_ImageProcessing_MASTER_HEADER ${DO_SOURCE_DIR}/ImageProcessing.hpp)
  # Header files
  file(GLOB DO_ImageProcessing_HEADER_FILES
       ${DO_ImageProcessing_SOURCE_DIR}/*.hpp)
  # Source files
  file(GLOB DO_ImageProcessing_SOURCE_FILES
       ${DO_ImageProcessing_SOURCE_DIR}/*.cpp)
  # All files here
  set(DO_ImageProcessing_HEADER_FILES
      ${DO_ImageProcessing_MASTER_HEADER}
      ${DO_ImageProcessing_HEADER_FILES})
  
  # Custom library project organization.
  source_group("0. Master Header File" FILES
               ${DO_ImageProcessing_MASTER_HEADER})  
  source_group("1. Linear Filtering" FILES
               ${DO_ImageProcessing_SOURCE_DIR}/LinearFiltering.hpp)
  source_group("2. Deriche IIR Filters" FILES
               ${DO_ImageProcessing_SOURCE_DIR}/Deriche.hpp)
  source_group("3. Differential Calculus, ..." FILES
               ${DO_ImageProcessing_SOURCE_DIR}/Differential.hpp
               ${DO_ImageProcessing_SOURCE_DIR}/Determinant.hpp
               ${DO_ImageProcessing_SOURCE_DIR}/Norm.hpp
               ${DO_ImageProcessing_SOURCE_DIR}/Orientation.hpp
               ${DO_ImageProcessing_SOURCE_DIR}/SecondMomentMatrix.hpp)
  source_group("4. Interpolation" FILES
               ${DO_ImageProcessing_SOURCE_DIR}/Interpolation.hpp)
  source_group("5. Reduce, Enlarge, Warp..." FILES
               ${DO_ImageProcessing_SOURCE_DIR}/Scaling.hpp
               ${DO_ImageProcessing_SOURCE_DIR}/Warp.hpp)
  source_group("6. Gaussian Scale-Space" FILES
               ${DO_ImageProcessing_SOURCE_DIR}/ImagePyramid.hpp
               ${DO_ImageProcessing_SOURCE_DIR}/DoG.hpp
               ${DO_ImageProcessing_SOURCE_DIR}/Extrema.hpp)
endmacro (do_list_imageprocessing_source_files)

macro (do_load_packages_for_imageprocessing_library)
  include(${DO_Core_USE_FILE})
  if (NOT APPLE)
    find_package(OpenMP)
    if (OPENMP_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
  endif()
endmacro (do_load_packages_for_imageprocessing_library)

macro (do_create_variables_for_imageprocessing_library)
  set(DO_ImageProcessing_LIBRARIES "")
  set(DO_ImageProcessing_LINK_LIBRARIES ${DO_ImageProcessing_LIBRARIES})
endmacro (do_create_variables_for_imageprocessing_library)


do_load_packages_for_imageprocessing_library()


if (DO_USE_FROM_SOURCE)
  get_property(DO_ImageProcessing_ADDED
               GLOBAL PROPERTY _DO_ImageProcessing_INCLUDED)
  if (NOT DO_ImageProcessing_ADDED)
    do_set_imageprocessing_source_dir()
    do_list_imageprocessing_source_files()
    do_create_variables_for_imageprocessing_library()
    
    # Static library
    do_append_library(
      ImageProcessing STATIC
      "${DO_SOURCE_DIR}"
      "${DO_ImageProcessing_HEADER_FILES}"
      "${DO_ImageProcessing_SOURCE_FILES}"
      "${DO_ImageProcessing_LINK_LIBRARIES}"
      )

    # Shared library
    if (DO_BUILD_SHARED_LIBS)
      do_append_library(
        ImageProcessing_SHARED SHARED
        "${DO_SOURCE_DIR}"
        "${DO_ImageProcessing_HEADER_FILES}"
        "${DO_ImageProcessing_SOURCE_FILES}"
        "${DO_ImageProcessing_LINK_LIBRARIES}"
        )
    endif ()
  endif ()
endif ()

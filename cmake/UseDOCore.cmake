macro (do_set_core_source_dir)
  set(DO_Core_SOURCE_DIR ${DO_SOURCE_DIR}/Core)
endmacro()

macro (do_list_core_source_files)
  # Master header file
  set(DO_Core_MASTER_HEADER ${DO_SOURCE_DIR}/Core.hpp)
  source_group("Master Header File" FILES ${DO_Core_MASTER_HEADER})
  # All files here
  file(GLOB DO_Core_HEADER_FILES FILES ${DO_Core_SOURCE_DIR}/*.hpp)
  list(APPEND DO_Core_HEADER_FILES ${DO_Core_MASTER_HEADER})
  file(GLOB DO_Core_SOURCE_FILES FILES ${DO_Core_SOURCE_DIR}/*.cpp)
  
  # To customize the project contents.
	# Linear algebra by integrating the 'Eigen' library
  set(DO_Core_EIGEN
      ${DO_Core_SOURCE_DIR}/EigenExtension.hpp)
  source_group("Eigen Integration" FILES ${DO_Core_EIGEN})
  # Template meta-programming stuff
  set(DO_Core_METAPROGRAMMING
      ${DO_Core_SOURCE_DIR}/StaticAssert.hpp
      ${DO_Core_SOURCE_DIR}/Meta.hpp)
  source_group("Template Meta-Programming" FILES ${DO_Core_METAPROGRAMMING})
  # N-dimensional array with N-dimensional iterators
  set(DO_Core_MULTIARRAY
      ${DO_Core_SOURCE_DIR}/ArrayIterators.hpp
      ${DO_Core_SOURCE_DIR}/MultiArray.hpp
      ${DO_Core_SOURCE_DIR}/SparseMultiArray.hpp)
  source_group("Multi-Array" FILES ${DO_Core_MULTIARRAY})
  # Image and color data structures
  set(DO_Core_IMAGE
      ${DO_Core_SOURCE_DIR}/Color.hpp
      ${DO_Core_SOURCE_DIR}/Image.hpp
			${DO_Core_SOURCE_DIR}/Subimage.hpp)
  source_group("Image and Color" FILES ${DO_Core_IMAGE})
  # Tree data structure
  set(DO_Core_TREE
      ${DO_Core_SOURCE_DIR}/Tree.hpp)
  source_group("Tree" FILES ${DO_Core_TREE})
  set(DO_Core_TIMER
      ${DO_Core_SOURCE_DIR}/Timer.hpp
      ${DO_Core_SOURCE_DIR}/Timer.cpp)
  source_group("Timer" FILES ${DO_Core_TIMER})
endmacro (do_list_core_source_files)

macro (do_create_variables_for_core_library)
  set(DO_Core_LIBRARIES DO_Core)
  set(DO_Core_LINK_LIBRARIES "")
endmacro (do_create_variables_for_core_library)

include_directories(${Eigen3_DIR} ${DO_INCLUDE_DIR})

if (DO_USE_FROM_SOURCE)
  get_property(DO_Core_ADDED GLOBAL PROPERTY _DO_Core_INCLUDED)
  if (NOT DO_Core_ADDED)
    do_set_core_source_dir()
    do_list_core_source_files()
    do_create_variables_for_core_library()
    do_generate_library("Core")
  endif ()
endif ()

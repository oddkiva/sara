# include directories
# link dependencies
# include files
# source files

# Geometry depends of Core and Graphics
do_include_internal_dirs("Core;Graphics")

if (DO_USE_FROM_SOURCE)
  get_property(DO_Geometry_ADDED GLOBAL PROPERTY _DO_Geometry_INCLUDED)
  if (NOT DO_Geometry_ADDED)
    do_glob_directory(${DO_SOURCE_DIR}/Geometry)
    do_create_common_variables("Geometry")
    do_set_internal_dependencies("Geometry" "Graphics")
    do_generate_library("Geometry")
  endif ()
endif ()

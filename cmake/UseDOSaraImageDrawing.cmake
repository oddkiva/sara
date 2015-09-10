include_directories(
  ${DO_Sara_INCLUDE_DIR}
  ${DO_Sara_INCLUDE_DIR}/antigrain/include
  ${DO_Sara_ThirdParty_DIR}/eigen
)

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageDrawing_ADDED GLOBAL PROPERTY _DO_Sara_ImageDrawing_INCLUDED)
  if (NOT DO_Sara_ImageDrawing_ADDED)
    do_glob_directory(${DO_Sara_SOURCE_DIR}/ImageDrawing)
    do_create_common_variables("ImageDrawing")
    do_generate_library("ImageDrawing")
    target_link_libraries(DO_Sara_ImageDrawing antigrain)
  endif ()
endif ()

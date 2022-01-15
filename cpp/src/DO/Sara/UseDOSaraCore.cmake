get_property(DO_Sara_Core_ADDED GLOBAL PROPERTY _DO_Sara_Core_INCLUDED)

if(NOT DO_Sara_Core_ADDED)
  sara_glob_directory(${DO_Sara_SOURCE_DIR}/Core)
  sara_create_common_variables("Core")
  sara_generate_library("Core")

  if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
    target_include_directories(DO_Sara_Core PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(DO_Sara_Core PUBLIC ${HDF5_CXX_LIBRARIES})
    target_compile_definitions(DO_Sara_Core
      PUBLIC
      $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING>
    )
  endif()
endif()

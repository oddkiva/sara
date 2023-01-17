get_property(DO_Sara_RANSAC_ADDED GLOBAL PROPERTY _DO_Sara_RANSAC_INCLUDED)

if(NOT DO_Sara_RANSAC_ADDED)
  sara_glob_directory(${DO_Sara_SOURCE_DIR}/RANSAC)
  sara_create_common_variables("RANSAC")
  sara_generate_library("RANSAC")

  target_include_directories(
    DO_Sara_RANSAC #
    PUBLIC ${CMAKE_SOURCE_DIR}/cpp/third-party/eigen)
  target_compile_features(DO_Sara_RANSAC INTERFACE cxx_std_20)
endif()

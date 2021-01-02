project(DO_Sara_SfM_Pipelines)

message(STATUS "  - DO_Sara_SfM_Pipelines")

find_package(DO_Sara
  COMPONENTS Graphics ImageIO FileSystem FeatureMatching MultiViewGeometry SfM
  REQUIRED)


add_custom_target(sfm_pipeline_helper ALL)
add_custom_command(TARGET sfm_pipeline_helper
  POST_BUILD
  DEPENDS sfm_pipeline.py
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/sfm_pipeline.py
          ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMENT "Copy the Python helper program to use the SfM pipeline")
set_property(TARGET sfm_pipeline_helper
  PROPERTY FOLDER "Pipelines/Sara/SfM")


file(GLOB pipelines_SOURCE_FILES FILES *.cpp)

foreach (file ${pipelines_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${file})
  target_include_directories(${filename}
    PRIVATE
    ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${filename}
    PRIVATE
    ${DO_Sara_LIBRARIES}
    Boost::program_options
    Boost::filesystem
    ${HDF5_LIBRARIES})
  target_compile_definitions(${filename}
    PRIVATE
    ${SARA_DEFINITIONS}  # so that the `src_path()` macro becomes valid.
    BOOST_ALL_DYN_LINK
    BOOST_ALL_NO_LIB)
  set_property(
      TARGET ${filename}
      PROPERTY FOLDER "Pipelines/Sara/SfM")
endforeach ()

project(DO_Sara_SfM_Pipelines)

message(STATUS "  - DO_Sara_SfM_Pipelines")

#set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS filesystem system program_options REQUIRED)
find_package(HDF5 COMPONENTS CXX REQUIRED)

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
  target_include_directories(${filename} PRIVATE
    ${Boost_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${filename} PRIVATE
    ${DO_Sara_LIBRARIES}
    ${Boost_LIBRARIES}
    ${HDF5_LIBRARIES})
  target_compile_definitions(${filename} PRIVATE
    ${SARA_DEFINITIONS}  # so that the `src_path()` macro becomes valid.
    -DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
  set_property(
      TARGET ${filename}
      PROPERTY FOLDER "Pipelines/Sara/SfM")
endforeach ()

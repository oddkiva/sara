project(DO_Pipelines)

message(STATUS "  - DO_Pipelines")

if (UNIX)
  find_package(HDF5 COMPONENTS CXX REQUIRED)
endif ()

find_package(DO_Sara
  COMPONENTS Graphics ImageIO FileSystem FeatureMatching MultiViewGeometry SfM
  REQUIRED)

file(GLOB pipelines_SOURCE_FILES FILES *.cpp)

foreach (file ${pipelines_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${file})
  target_link_libraries(${filename} ${DO_Sara_LIBRARIES})
endforeach ()

target_include_directories(detect_sift PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(detect_sift ${DO_Sara_LIBRARIES} ${HDF5_LIBRARIES})

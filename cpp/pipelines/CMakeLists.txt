project(DO_Pipelines)

message(STATUS "  - DO_Pipelines")

#set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS filesystem system program_options REQUIRED)
find_package(HDF5 COMPONENTS CXX REQUIRED)

find_package(DO_Sara
  COMPONENTS Graphics ImageIO FileSystem FeatureMatching MultiViewGeometry SfM
  REQUIRED)

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
    -DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
  set_property(
      TARGET ${_test_name}
      PROPERTY FOLDER "DO Sara Pipelines/SfM")
endforeach ()

if (NOT SARA_BUILD_VIDEOIO)
  return ()
endif ()

find_package(DO_Sara COMPONENTS Core Graphics ImageProcessing ImageIO VideoIO REQUIRED)

file(GLOB taskflow_SOURCE_FILES FILES *.cpp)

foreach (file ${taskflow_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${filename}.cpp)
  target_include_directories(${filename}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/third-party/taskflow/include)
  target_link_libraries(${filename}
    PRIVATE
    ${DO_Sara_LIBRARIES}
    $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  set_target_properties(${filename}
      PROPERTIES
      COMPILE_FLAGS ${SARA_DEFINITIONS}
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set_property(TARGET ${filename}
    PROPERTY
    FOLDER "Examples/Sara/Taskflow")
endforeach ()

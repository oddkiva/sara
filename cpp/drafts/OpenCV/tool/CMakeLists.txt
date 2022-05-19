file(GLOB SOURCE_FILES FILES *.cpp)

foreach(file ${SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  add_executable(${filename} ${filename}.cpp)
  target_include_directories(${filename} PRIVATE ${CERES_INCLUDE_DIRS})
  target_link_libraries(
    ${filename}
    PRIVATE
    DO::Sara::Core DO::Sara::VideoIO DO::Sara::Graphics #
    ${OpenCV_LIBS} #
    ${CERES_LIBRARIES}
    HomographyUtilities
    $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  set_target_properties(
    ${filename} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set_property(TARGET ${filename} PROPERTY FOLDER "DRAFTS/OpenCV")
endforeach()

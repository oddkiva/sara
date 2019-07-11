project(DO_Kalpana_Examples)

message(STATUS "  - DO_Kalpana_Examples")

find_package(OpenGL REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets OpenGL OpenGLExtensions REQUIRED)

add_definitions(${Qt5Widgets_DEFINITIONS})
if (UNIX)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
endif ()
add_definitions(-DGL_GLEXT_PROTOTYPES)

file(GLOB examples_SRC_FILES FILES *.cpp  *.hpp)
foreach (file ${examples_SRC_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  kalpana_add_example(${filename} ${file})
  target_link_libraries(
    ${filename}
    DO_Kalpana_2D DO_Kalpana_3D DO_Kalpana_Math
    Qt5::Widgets Qt5::OpenGL Qt5::OpenGLExtensions)
endforeach()

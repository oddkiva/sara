project(DO_Kalpana_Examples)

message(STATUS "  - DO_Kalpana_Examples")

find_package(OpenGL REQUIRED)
find_package(Qt5 COMPONENTS Core Widgets OpenGL OpenGLExtensions REQUIRED)
find_package(DO_Sara COMPONENTS Core REQUIRED)

file(GLOB examples_SRC_FILES FILES *.cpp)
foreach (file ${examples_SRC_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  kalpana_add_example(${filename} ${file})

  target_include_directories(${filename}
    PRIVATE ${HDF5_INCLUDE_DIRS})

  target_link_libraries(${filename}
    PRIVATE
    ${DO_Sara_LIBRARIES}
    ${HDF5_LIBRARIES}
    DO_Kalpana_2D DO_Kalpana_3D DO_Kalpana_Math
    Qt5::Widgets Qt5::OpenGL Qt5::OpenGLExtensions)

  target_compile_definitions(${filename}
    PRIVATE
    $<$<PLATFORM_ID:Darwin>:GL_SILENCE_DEPRECATION>
    ${SARA_DEFINITIONS})

  if (NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${filename}
      PRIVATE
      DO_KALPANA_STATIC)
  endif ()
endforeach()

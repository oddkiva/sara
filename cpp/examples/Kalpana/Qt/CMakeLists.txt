find_package(
  DO_Sara
  COMPONENTS Core
  REQUIRED)

file(GLOB examples_SRC_FILES FILES *.cpp)
foreach(file ${examples_SRC_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  add_executable(${filename} ${file})

  target_include_directories(${filename} PRIVATE ${HDF5_INCLUDE_DIRS})

  target_link_libraries(
    ${filename}
    PRIVATE ${DO_Sara_LIBRARIES}
            ${HDF5_LIBRARIES}
            DO::Kalpana::Qt::2D #
            DO::Kalpana::Qt::3D #
            Qt${QT_VERSION}::Widgets
            Qt${QT_VERSION}::OpenGL
            $<$<EQUAL:${QT_VERSION},6>:Qt6::OpenGLWidgets>)

  target_compile_definitions(
    ${filename} PRIVATE $<$<PLATFORM_ID:Darwin>:GL_SILENCE_DEPRECATION>
                        ${SARA_DEFINITIONS})

  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${filename} PRIVATE DO_KALPANA_STATIC)
  endif()

  set_property(TARGET ${filename} PROPERTY FOLDER "Examples/Kalpana/Qt")
endforeach()

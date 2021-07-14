set(DO_Kalpana_VERSION 0.1.0)
set(DO_Kalpana_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(DO_Kalpana_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})


set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include(kalpana_macros)

find_package(DO_Sara COMPONENTS Core REQUIRED)

if (Qt${QT_VERSION}_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

kalpana_dissect_version(${DO_Kalpana_VERSION})


# Create the following libraries.
set(DO_Kalpana_LIBRARIES 2D 3D Math Plot)
foreach (library ${DO_Kalpana_LIBRARIES})
  kalpana_glob_directory(${DO_Kalpana_SOURCE_DIR}/${library})
  kalpana_create_common_variables("${library}")
  kalpana_generate_library("${library}")

  if (NOT ${library} STREQUAL "Plot")
    target_link_libraries(DO_Kalpana_${library}
      Qt${QT_VERSION}::Core
      Qt${QT_VERSION}::Widgets
      Qt${QT_VERSION}::OpenGL
      Qt${QT_VERSION}::OpenGLWidgets
      # Qt${QT_VERSION}::OpenGLExtensions
      ${OPENGL_LIBRARIES}
      ${DO_Sara_LIBRARIES})
  endif ()
endforeach ()

target_link_libraries(DO_Kalpana_3D DO_Kalpana_Math)
target_compile_definitions(DO_Kalpana_3D PRIVATE -DGL_SILENCE_DEPRECATION)

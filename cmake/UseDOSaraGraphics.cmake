find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)
find_package(OpenGL REQUIRED)

add_definitions(${Qt5Widgets_DEFINITIONS})
if (UNIX)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
endif ()

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Graphics_ADDED GLOBAL PROPERTY _DO_Sara_Graphics_INCLUDED)
  if (NOT DO_Sara_Graphics_ADDED)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Graphics)
    sara_create_common_variables("Graphics")
    sara_generate_library("Graphics")

    target_include_directories(DO_Sara_Graphics
      PUBLIC
      ${Qt5Widgets_INCLUDE_DIRS}
      ${Qt5OpenGL_INCLUDE_DIRS}
      ${DO_Sara_INCLUDE_DIR})
    target_link_libraries(DO_Sara_Graphics
      PUBLIC
      Qt5::Widgets Qt5::OpenGL
      ${OPENGL_LIBRARIES})
    target_compile_definitions(DO_Sara_Graphics
      PRIVATE
      -DGL_SILENCE_DEPRECATION)

    if (WIN32)
      target_link_libraries(DO_Sara_Graphics PRIVATE Qt5::WinMain)
    endif ()


    set(CMAKE_AUTOMOC OFF)
    set(CMAKE_INCLUDE_CURRENT_DIR OFF)
  endif()
endif ()

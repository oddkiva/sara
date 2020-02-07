if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Graphics_ADDED GLOBAL PROPERTY _DO_Sara_Graphics_INCLUDED)
  if (NOT DO_Sara_Graphics_ADDED)
    find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)
    find_package(OpenGL REQUIRED)

    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Graphics)
    sara_create_common_variables("Graphics")
    sara_generate_library("Graphics")

    target_include_directories(DO_Sara_Graphics
      PUBLIC
      ${DO_Sara_INCLUDE_DIR})

    target_link_libraries(DO_Sara_Graphics
      PUBLIC
      Qt5::Widgets
      Qt5::OpenGL
      ${OPENGL_LIBRARIES})

    if (WIN32)
      target_link_libraries(DO_Sara_Graphics PRIVATE Qt5::WinMain)
    endif ()

    target_compile_definitions(DO_Sara_Graphics PRIVATE GL_SILENCE_DEPRECATION)

    set(CMAKE_AUTOMOC OFF)
    set(CMAKE_INCLUDE_CURRENT_DIR OFF)
  endif()
endif ()

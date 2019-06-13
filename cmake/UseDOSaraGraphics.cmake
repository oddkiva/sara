if (WIN32)
  # Temporary workaround for windows 8
  list(APPEND CMAKE_PREFIX_PATH
    "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x64")
endif ()

if (DEFINED ENV{QTDIR})
  message(STATUS
    "Found environment variable QTDIR = $ENV{QTDIR} and appending "
    "it to CMAKE_MODULE_PATH")
  list(APPEND CMAKE_PREFIX_PATH $ENV{QTDIR})
endif ()

find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)
find_package(OpenGL REQUIRED)

add_definitions(${Qt5Widgets_DEFINITIONS})
if (UNIX)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
endif ()

include_directories(
  ${Qt5Widgets_INCLUDE_DIRS}
  ${Qt5OpenGL_INCLUDE_DIRS}
  ${DO_Sara_INCLUDE_DIR})

if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_Graphics_ADDED GLOBAL PROPERTY _DO_Sara_Graphics_INCLUDED)
  if (NOT DO_Sara_Graphics_ADDED)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_INCLUDE_CURRENT_DIR ON)

    sara_glob_directory(${DO_Sara_SOURCE_DIR}/Graphics)
    sara_create_common_variables("Graphics")
    sara_generate_library("Graphics")
    target_link_libraries(
      DO_Sara_Graphics
      Qt5::Widgets Qt5::OpenGL ${OPENGL_LIBRARIES})

    if (WIN32)
      target_link_libraries(DO_Sara_Graphics Qt5::WinMain)
    endif ()

    set(CMAKE_AUTOMOC OFF)
    set(CMAKE_INCLUDE_CURRENT_DIR OFF)
  endif()
endif ()

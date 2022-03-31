sara_step_message("Found ${CMAKE_CXX_COMPILER_ID} compiler:")


# By default, use the math constants defined in <cmath> header.
add_definitions(-D_USE_MATH_DEFINES)

# Visual C++ compiler
if (CMAKE_COMPILER_IS_GNUCXX)
  sara_substep_message(
    "${CMAKE_CXX_COMPILER_ID} compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=72")
  # Turn off Eigen warnings.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")

  # Enable colors in gcc log output.
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 4.8)
    sara_substep_message("Enable colored output of GCC.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
  endif ()
endif ()

if (UNIX)
  # Base compilation flags.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
  endif()
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

  # Additional flags for Release builds.
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  # Additional flags for Debug builds to code coverage.
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -D_DEBUG -fno-inline")
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
  endif ()
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL Emscripten)
  # Support exceptions.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fexceptions")

  # Silence Eigen compile warnings.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy-with-user-provided-copy")

  # Additional flags for Release builds.
  set(CMAKE_CXX_FLAGS_RELEASE "SHELL:-O3")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "SHELL:-O2")
  # Additional flags for Debug builds to code coverage.
  set(CMAKE_CXX_FLAGS_DEBUG "SHELL:-O0 -DDEBUG -D_DEBUG -fno-inline")
endif ()


# Generate position independent code for static libraries.
# (cf. EasyExif third-party library)
if (SARA_BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

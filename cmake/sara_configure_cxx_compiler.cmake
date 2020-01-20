sara_step_message("Found ${CMAKE_CXX_COMPILER_ID} compiler:")

# By default, use the math constants defined in <cmath> header.
add_definitions(-D_USE_MATH_DEFINES)

# Drop older compiler support in favor of C++17... I know it may be a
# controversial decision.
set(CMAKE_CXX_STANDARD 17)

# Visual C++ compiler
if (MSVC)
  add_definitions(
    /D_SCL_SECURE_NO_WARNINGS
    /D_CRT_SECURE_NO_DEPRECATE
    /D_SILENCE_CXX17_ADAPTOR_TYPEDEFS_DEPRECATION_WARNING  # Eigen
    /bigobj
    /wd4251)
  message(STATUS "  - Disabled annoying warnings in MSVC.")

  add_definitions(/EHsc)
  message(STATUS
          "  - Using /EHsc: catches C++ exceptions only and tells the "
          "compiler to assume that extern C functions never throw a C++ "
          "exception.")

  if (MSVC_VERSION EQUAL 1700)
    message(STATUS
            "  - Using version 2012: setting '_VARIADIC_MAX=10' to compile "
            "'Google Test'")
    add_definitions(/D_VARIADIC_MAX=10)
  endif ()

# GNU compiler
elseif (CMAKE_COMPILER_IS_GNUCXX)
  sara_substep_message(
    "${CMAKE_CXX_COMPILER_ID} compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=72")

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")
  if (UNIX AND SARA_BUILD_SHARED_LIBS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
  endif ()

  #Additional flags for Release builds.
  set(CMAKE_CXX_RELEASE_FLAGS "-03 -ffast-math")
  # Additional flags for Debug builds, which include code coverage.
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG -D_DEBUG -fno-inline")
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
  endif ()
endif ()

# Activate OpenMP by default.
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

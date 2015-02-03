do_step_message("Found ${CMAKE_CXX_COMPILER_ID} compiler:")

# Visual C++ compiler
if (MSVC)
  add_definitions(/D_SCL_SECURE_NO_WARNINGS /D_CRT_SECURE_NO_DEPRECATE)
  message(STATUS "  - NON-SECURE warnings are disabled.")
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

# Clang compiler
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  endif ()
  set (ENABLE_CXX11 "-std=c++11")

# GNU compiler
elseif (CMAKE_COMPILER_IS_GNUCXX)               
  exec_program(${CMAKE_C_COMPILER}
               ARGS "-dumpversion"
               OUTPUT_VARIABLE _gcc_version_info)
  string(REGEX REPLACE "^([0-9]+).*$" "\\1" 
         GCC_MAJOR ${_gcc_version_info})
  string(REGEX REPLACE
         "^[0-9]+\\.([0-9]+).*$" "\\1"
         GCC_MINOR ${_gcc_version_info})
  string(REGEX REPLACE
         "^[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1"
         GCC_PATCH ${_gcc_version_info})
  if(GCC_PATCH MATCHES "\\.+")
    set(GCC_PATCH "")
  endif()
  if(GCC_MINOR MATCHES "\\.+")
    set(GCC_MINOR "")
  endif()
  if(GCC_MAJOR MATCHES "\\.+")
    set(GCC_MAJOR "")
  endif()
  set(GCC_VERSION "${GCC_MAJOR}.${GCC_MINOR}")
  
  do_substep_message(
    "${CMAKE_CXX_COMPILER_ID} compiler version: ${GCC_VERSION}")
  if (NOT GCC_VERSION  VERSION_LESS 4.5 AND GCC_VERSION  VERSION_LESS 4.7)
    set(ENABLE_CXX11 "-std=c++0x")
  else ()
    set (ENABLE_CXX11 "-std=c++11")
  endif ()
endif ()

if (UNIX)
  # Base compilation flags.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wundef")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunused-variable")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")
  if (DEFINED ENABLE_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ENABLE_CXX11}")
  endif ()
  # Additional flags for Release builds.
  set(CMAKE_CXX_RELEASE_FLAGS "-03 -ffast-math")
  # Additional flags for Debug builds, which include code coverage.
  set(CMAKE_CXX_FLAGS_DEBUG
      "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG -D_DEBUG -fno-inline")
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
  endif ()
endif ()

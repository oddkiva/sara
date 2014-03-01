# IMPORTANT NOTE:
#
# I have decided DO-CV will work on recent C++ compilers because C++11 features
# are used. Code is much more elegant, more concise, easier to maintain.
#
# C++11 used features:
# - auto
# - lambda
#

do_step_message("Found ${CMAKE_CXX_COMPILER_ID} compiler:")

# Visual C++ compiler
if (MSVC)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
  message(STATUS "  - NON-SECURE warnings are disabled.")
  add_definitions(/EHsc)
  message(STATUS "  - Using /EHsc: catches C++ exceptions only and tells the compiler to assume that extern C functions never throw a C++ exception.")
  if (MSVC_VERSION EQUAL 1700)
    message(STATUS "  - Using version 2012: setting '_VARIADIC_MAX=10' to compile 'Google Test'")
    add_definitions(/D _VARIADIC_MAX=10)
  endif ()

# Clang compiler
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set (ENABLE_CXX11 "-std=c++11 -stdlib=libc++")

# GNU compiler
elseif (CMAKE_COMPILER_IS_GNUCXX)
  execute_process(COMMAND "${CMAKE_CXX_COMPILER} -dumpversion"
    OUTPUT_VARIABLE GCC_VERSION)
  if (GCC_VERSION VERSION_LESS 4.5)
    message(FATAL_ERROR "GNU compiler version lower than 4.5 are not supported anymore: C++0x features (auto and lambda) are needed.")
  elseif (GCC_VERSION VERSION_LESS 4.7)
    set(ENABLE_CXX11 "-std=c++0x")
  else ()
    set (ENABLE_CXX11 "-std=c++11")
  endif ()
else ()
  message("WARNING: Compiler '${CMAKE_CXX_COMPILER}' may not be supported by DO-CV. Make sure that C++0x features are needed (auto and lambda) and adjust the CMake variable 'ENABLE_CXX11'. Otherwise, report back to me: david.ok8@gmail.com and I'll try to do what I can.")
endif ()

do_step_message("Activating C++11 features with: '${ENABLE_CXX11}'")

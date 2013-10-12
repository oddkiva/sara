cmake_minimum_required(VERSION 2.6)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(flann)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)

include(${PROJECT_SOURCE_DIR}/cmake/flann_utils.cmake)
set(FLANN_VERSION 1.8.4)
DISSECT_VERSION()
GET_OS_INFO()

################################################################################
# CMAKE_C_COMPILER_ID and CMAKE_CXX_COMPILER_ID are documented as internal 
# variables subject to change at: http://www.cmake.org/cmake/help/v2.8.8/cmake.html#variable:CMAKE_LANG_COMPILER_ID
# However, currently this seems the most robust way to detect the presence of 
# the Clang compiler.
# 
if("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_COMPILER_IS_CLANG 1)
endif ()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_COMPILER_IS_CLANGXX 1)
endif ()


################################################################################
# Build options: OpenMP and CUDA.
# 
option(USE_OPENMP "Use OpenMP multi-threading" ON)
option(BUILD_CUDA_LIB "Build CUDA library" OFF)

set(NVCC_COMPILER_BINDIR "" CACHE PATH  "Directory where nvcc should look for C++ compiler. This is passed to nvcc through the --compiler-bindir option.")

################################################################################
# OpenMP
# 
if (USE_OPENMP AND NOT APPLE)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(WARNING "OpenMP NOT found")
    endif()
endif()

################################################################################
# CUDA support
# 
if (BUILD_CUDA_LIB)
    find_package(CUDA)
    if (CUDA_FOUND)
        message(STATUS "CUDA found (include: ${CUDA_INCLUDE_DIRS}, lib: ${CUDA_LIBRARIES})")
        include_directories(${CUDA_INCLUDE_DIRS})
    else(CUDA_FOUND)
        message(STATUS "CUDA not found, CUDA library will not be built")
        set(BUILD_CUDA_LIB OFF)
    endif(CUDA_FOUND)
endif(BUILD_CUDA_LIB)


################################################################################
# Compiler specific flags
# 
if (WIN32)
    if (MSVC)
        add_definitions("/W1 -D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS /bigobj")
        add_definitions("/wd4244 /wd4291 /wd4267")
    endif (MSVC)
else(WIN32)
    add_definitions( "-Wall -Wno-unknown-pragmas -Wno-unused-function" )
endif(WIN32)

################################################################################
# Add the project
# 
include_directories(${PROJECT_SOURCE_DIR}/src/cpp)
add_subdirectory( src )

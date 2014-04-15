cmake_minimum_required(VERSION 2.6)

project(DO_MatchPropagation)

if (POLICY CMP0020)
  cmake_policy(SET CMP0020 OLD)
endif (POLICY CMP0020)

# ==============================================================================
# Find DO++
find_package(DO COMPONENTS 
             Core Graphics KDTree Geometry FileSystem 
             Features FeatureDetectors Match
             FeatureMatching REQUIRED)
include(${DO_USE_FILE})

# =============================================================================
# Find OpenMP
if (NOT APPLE)
  find_package(OpenMP) 
  if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif ()

# ==============================================================================
# Generate the library and programs.
add_subdirectory(src)
add_subdirectory(test)
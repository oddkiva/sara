# Boost unit test framework.
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost
  COMPONENTS filesystem serialization system unit_test_framework
  REQUIRED)

# Visual leak detector.
if (WIN32 AND SARA_USE_VLD)
  find_package(VisualLeakDetector)
  include_directories(${VLD_INCLUDE_DIR})
  link_directories(${VLD_LINK_DIRECTORIES})
endif ()


add_subdirectory(Sara)

if (CUDA_FOUND)
  sara_message("Building DO::Shakti Libraries")
  add_subdirectory(Shakti)
else ()
  sara_message("Cannot use DO::Shakti libraries!")
endif ()

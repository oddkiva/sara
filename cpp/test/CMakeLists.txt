# Boost unit test framework.
if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(Boost_INCLUDE_DIR
    /Users/david/GitHub/Apple-Boost-BuildScript/dist/boost.xcframework/Headers)
  set(Boost_LIBRARIES
    /Users/david/GitHub/Apple-Boost-BuildScript/dist/boost.xcframework/ios-arm64/libboost.a)
	find_package(XCTest REQUIRED)
else ()
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  find_package(Boost
    COMPONENTS filesystem serialization system unit_test_framework
    REQUIRED)
endif ()

# add_subdirectory(BoostToXCTest)

# Visual leak detector.
if (WIN32 AND SARA_USE_VLD)
  find_package(VisualLeakDetector)
  include_directories(${VLD_INCLUDE_DIR})
  link_directories(${VLD_LINK_DIRECTORIES})
endif ()


add_subdirectory(Sara)

if (CMAKE_CUDA_COMPILER)
  sara_message("Building DO::Shakti Libraries")
  add_subdirectory(Shakti)
else ()
  sara_message("Cannot use DO::Shakti libraries!")
endif ()

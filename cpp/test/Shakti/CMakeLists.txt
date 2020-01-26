include_directories(${DO_Sara_ThirdParty_DIR}/gtest/include)

message(STATUS "")
shakti_step_message("Appending test projects:")

if (WIN32 AND DO_USE_VLD)
  find_package(VisualLeakDetector)
  include_directories(${VLD_INCLUDE_DIR})
  link_directories(${VLD_LINK_DIRECTORIES})
endif ()

add_subdirectory(ImageProcessing)
add_subdirectory(MultiArray)
add_subdirectory(Segmentation)

if (NOT OpenCL_FOUND)
  return ()
endif ()

find_package(DO_Sara COMPONENTS Core ImageProcessing ImageIO REQUIRED)

add_subdirectory(Core)

if (SARA_BUILD_TESTS)
  add_subdirectory(test)
endif ()
add_subdirectory(examples)

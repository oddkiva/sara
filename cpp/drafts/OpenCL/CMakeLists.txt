find_package(Boost COMPONENTS
  filesystem serialization system unit_test_framework REQUIRED)
find_package(OpenCL REQUIRED)
find_package(OpenGL REQUIRED)
if (NOT APPLE)
  find_package(GLEW REQUIRED)
endif ()
find_package(glfw3 REQUIRED)
find_package(DO_Sara COMPONENTS Core ImageIO REQUIRED)


add_subdirectory(Core)
add_subdirectory(GL)

add_subdirectory(test)
add_subdirectory(examples)

# ============================================================================ #
# Compile the library.
include_directories(${Boost_INCLUDE_DIRS})

if (SARA_BUILD_VIDEOIO)
  add_definitions(-DPYSARA_BUILD_VIDEOIO)
endif ()

file(GLOB pysara_SRC_FILES FILES *.cpp)
if (NOT SARA_BUILD_VIDEOIO)
  list(REMOVE_ITEM pysara_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/VideoIO.cpp")
endif ()

add_library(pysara_pybind11 MODULE ${pysara_SRC_FILES})
target_link_libraries(pysara_pybind11
  PRIVATE pybind11::module
          DO_Sara_DisjointSets
          DO_Sara_Geometry
          DO_Sara_ImageIO)
          # DO_Sara_Features
          # DO_Sara_SfM)
if (SARA_BUILD_VIDEOIO)
  target_link_libraries(pysara_pybind11 PRIVATE DO_Sara_VideoIO)
endif ()
set_target_properties(pysara_pybind11 PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  FOLDER "Python")



# ============================================================================ #
# Copy the Python bindings in the runtime output directory.
add_custom_target(copy_pysara_module)

# Copy pysara.[so, dll] to the runtime output directory.
set(pysara_LIBRARY_PATH $<TARGET_FILE:pysara_pybind11>)
add_custom_command(TARGET copy_pysara_module
  COMMAND ${CMAKE_COMMAND} -E
  copy ${pysara_LIBRARY_PATH} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(copy_pysara_module PROPERTIES FOLDER "Python")

add_dependencies(copy_pysara_module pysara_pybind11)


# ============================================================================ #
# Install the library in the currently active python environment.
if (SARA_BUILD_TESTS AND SARA_BUILD_PYTHON_BINDINGS)
  add_subdirectory(test)
endif ()

install(TARGETS pysara_pybind11
  RUNTIME DESTINATION bin COMPONENT Libraries
  LIBRARY DESTINATION lib COMPONENT Libraries
  ARCHIVE DESTINATION lib COMPONENT Libraries)

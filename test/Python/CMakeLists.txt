find_program(NOSETESTS_PATH nosetests)
if(NOT NOSETESTS_PATH)
  message(FATAL_ERROR "nosetests not found! Aborting...")
endif()


# Copy the unit tests in the library output directory.
add_custom_target(copy_python_unittests)

# Create python module 'bin/do/sara'.
add_custom_command(TARGET copy_python_unittests
  COMMAND ${CMAKE_COMMAND} -E
  make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/do)

add_custom_command(TARGET copy_python_unittests
  COMMAND ${CMAKE_COMMAND} -E
          touch ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/do/__init__.py)

# Copy sara.[dll, so] to 'bin/do/sara'.
get_target_property(SARA_NAME sara LOCATION)
message("SARA_FILEPATH = ${SARA_NAME}")
add_custom_command(TARGET copy_python_unittests
  COMMAND ${CMAKE_COMMAND} -E
          copy ${SARA_NAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/do)

# Copy the test files in 'bin' directory.
file(GLOB PYTHON_SRC_FILES FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
foreach(pysource ${PYTHON_SRC_FILES})
  add_custom_command(TARGET copy_python_unittests
    COMMAND ${CMAKE_COMMAND} -E
    copy ${pysource} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endforeach()


# Run the unit tests with nosetests.
add_custom_target(run_nosetests
  COMMAND ${NOSETESTS_PATH}
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  COMMENT "Running Python tests.")

# Make sure the python unit tests are copied first before running unit tests.
add_dependencies(run_nosetests copy_python_unittests)
add_dependencies(copy_python_unittests sara)

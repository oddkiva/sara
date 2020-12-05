# ============================================================================ #
# Find nosetests.
find_program(NOSETESTS_EXECUTABLE nosetests)
if(NOT NOSETESTS_EXECUTABLE)
  message(FATAL_ERROR "nosetests not found! Aborting...")
endif()


# ============================================================================ #
# Run the unit tests.
add_custom_target(pytest
  COMMAND ${CMAKE_COMMAND}
      -E env PYTHONPATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
      ${NOSETESTS_EXECUTABLE} --ipdb
                              --nocapture
                              --verbosity=2
                              --with-coverage
                              --cover-package=do
  WORKING_DIRECTORY ${DO_Sara_DIR}/python
  COMMENT "Running Python tests.")
set_target_properties(pytest PROPERTIES FOLDER "Python")


# ============================================================================ #
# Make sure the python unit tests are copied first before running unit tests.
add_dependencies(pytest copy_pysara_module)
add_dependencies(copy_pysara_module pysara_pybind11)

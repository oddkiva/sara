# ============================================================================ #
# Run the unit tests.

add_custom_target(pytest
  COMMAND ${CMAKE_COMMAND}
      -E env PYTHONPATH=$<TARGET_FILE_DIR:pysara_pybind11>
      coverage run -m pytest -s
  # COMMAND ${CMAKE_COMMAND}
  #     -E env PYTHONPATH=$<TARGET_FILE_DIR:pysara_pybind11>
  #     coverage report
  WORKING_DIRECTORY ${DO_Sara_DIR}/python
  COMMENT "Running Python tests.")
set_target_properties(pytest PROPERTIES FOLDER "Python")


# ============================================================================ #
# Make sure the python unit tests are copied first before running unit tests.
add_dependencies(pytest copy_pysara_module)
add_dependencies(copy_pysara_module pysara_pybind11)

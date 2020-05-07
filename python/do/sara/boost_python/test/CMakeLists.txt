# ============================================================================ #
# Find nosetests.
find_program(NOSETESTS_EXECUTABLE nosetests)
if(NOT NOSETESTS_EXECUTABLE)
  message(FATAL_ERROR "nosetests not found! Aborting...")
endif()


# ============================================================================ #
# Copy the Python bindings in the runtime output directory.
add_custom_target(copy_pysara_module)

# Create python module to the runtime output directory.
add_custom_command(TARGET copy_pysara_module
  COMMAND ${CMAKE_COMMAND} -E
  copy_directory ${DO_Sara_DIR}/python/do
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/do)

# Copy pysara.[so, dll] to the runtime output directory.
set(pysara_LIBRARY_PATH $<TARGET_FILE:pysara>)
add_custom_command(TARGET copy_pysara_module
  COMMAND ${CMAKE_COMMAND} -E
  copy ${pysara_LIBRARY_PATH} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(copy_pysara_module PROPERTIES FOLDER "Python")


# ============================================================================ #
# Run the unit tests.
add_custom_target(pytest
  COMMAND ${NOSETESTS_EXECUTABLE} --ipdb --nocapture --verbosity=2 --with-coverage --cover-package=do
  WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  COMMENT "Running Python tests.")
set_target_properties(pytest PROPERTIES FOLDER "Python")


# ============================================================================ #
# Make sure the python unit tests are copied first before running unit tests.
add_dependencies(pytest copy_pysara_module)
add_dependencies(copy_pysara_module pysara)

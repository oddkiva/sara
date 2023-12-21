find_package(pybind11 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(oddkiva/sara/pybind11)
add_subdirectory(oddkiva/shakti/pybind11)

if(SARA_BUILD_TESTS AND SARA_BUILD_PYTHON_BINDINGS)
  # Python test runner as a CMake target.
  add_custom_target(
    pytest
    COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pysara_pybind11>
      coverage run -m pytest -s
    WORKING_DIRECTORY ${DO_Sara_DIR}/python
    COMMENT "Running Python tests...")
  set_target_properties(pytest PROPERTIES FOLDER "Python")

  # Python code coverage as a CMake target.
  add_custom_target(
    pycoverage
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:pysara_pybind11> coverage report
    WORKING_DIRECTORY ${DO_Sara_DIR}/python
    COMMENT "Making Python coverage report...")
  set_target_properties(pytest PROPERTIES FOLDER "Python")

  # Dependency DAG:
  #
  # - pysara_pybind11 -> copy_pysara_module
  # - pyshakti_pybind11 -> copy_pyshakti_module
  add_dependencies(pytest copy_pysara_module)
  add_dependencies(pytest copy_pyshakti_module)
  # - pytest -> pycoverage
  add_dependencies(pycoverage pytest)

endif()

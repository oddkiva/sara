# Find the needed third-party libraries.
find_package(NumPy 1.7.1 REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)

if (PYTHON_VERSION_MAJOR STREQUAL 3)
  set(SARA_BOOST_PYTHON python-py${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
else ()
  set(SARA_BOOST_PYTHON python)
endif ()
find_package(Boost COMPONENTS ${SARA_BOOST_PYTHON} REQUIRED)


# Compile the library.
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

file(GLOB sara_SRC_FILES FILES *.cpp)

python_add_module(sara ${sara_SRC_FILES})

target_link_libraries(sara
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  DO_Sara_DisjointSets
  DO_Sara_ImageIO
  DO_Sara_VideoIO)


# Install the library in the currently active python environment.
execute_process(COMMAND python -c
  "from distutils.sysconfig import get_python_lib; print get_python_lib()"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(sara_INSTALL_DIR ${PYTHON_SITE_PACKAGES_DIR}/do)

message("Installing sara Python bindings to ${sara_INSTALL_DIR}")

install(TARGETS sara
        RUNTIME DESTINATION ${sara_INSTALL_DIR}
        ARCHIVE DESTINATION ${sara_INSTALL_DIR}
        LIBRARY DESTINATION ${sara_INSTALL_DIR})

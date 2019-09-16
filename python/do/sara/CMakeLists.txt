# ============================================================================ #
# Find the needed third-party libraries.
find_package(NumPy 1.7.1 REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)

if (NOT SARA_BOOST_PYTHON_VERSION AND PYTHON_VERSION_MAJOR STREQUAL 3)
  set(SARA_BOOST_PYTHON_VERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
endif ()

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost)
if (Boost_VERSION VERSION_LESS 1.63)
  find_package(Boost
    COMPONENTS python${SARA_BOOST_PYTHON_VERSION}
    REQUIRED)
else ()
  find_package(Boost
    COMPONENTS python${SARA_BOOST_PYTHON_VERSION}
               numpy${SARA_BOOST_PYTHON_VERSION}
    REQUIRED)
endif ()



# ============================================================================ #
# Compile the library.
include_directories(
  ${Boost_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS})

if (SARA_BUILD_VIDEOIO)
  add_definitions(-DPYSARA_BUILD_VIDEOIO)
endif ()

file(GLOB pysara_SRC_FILES FILES *.cpp)
if (NOT SARA_BUILD_VIDEOIO)
  list(REMOVE_ITEM pysara_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/VideoIO.cpp")
endif ()

python_add_module(pysara ${pysara_SRC_FILES})

target_link_libraries(pysara
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  DO_Sara_DisjointSets
  DO_Sara_Geometry
  DO_Sara_ImageIO)
if (SARA_BUILD_VIDEOIO)
  target_link_libraries(pysara DO_Sara_VideoIO)

  if (UNIX AND NOT APPLE)
    target_link_libraries(pysara rt)
  endif ()
endif ()

target_compile_definitions(DO_Sara_FileSystem
  PRIVATE -DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)

# ============================================================================ #
# Install the library in the currently active python environment.
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if (SARA_SELF_CONTAINED_INSTALLATION)
  set(PYTHON_VERSION ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR})
  set(SARA_PYTHON_INSTALL_DIR
    ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/site-packages)
else ()
  set(SARA_PYTHON_INSTALL_DIR ${PYTHON_SITE_PACKAGES_DIR})
endif ()
message("Installing sara Python bindings to ${SARA_PYTHON_INSTALL_DIR}")

install(TARGETS pysara
        RUNTIME DESTINATION ${SARA_PYTHON_INSTALL_DIR}
        ARCHIVE DESTINATION ${SARA_PYTHON_INSTALL_DIR}
        LIBRARY DESTINATION ${SARA_PYTHON_INSTALL_DIR})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
        DESTINATION ${SARA_PYTHON_INSTALL_DIR}/do
        FILES_MATCHING PATTERN "*.py")



if (SARA_BUILD_TESTS AND SARA_BUILD_PYTHON_BINDINGS)
  add_subdirectory(test)
endif ()

set(Python_ADDITIONAL_VERSIONS 3.4)
find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python-py34 REQUIRED)


include_directories(
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)


file(GLOB pysara_SRC_FILES FILES *.cpp)
python_add_module(sara ${pysara_SRC_FILES})
target_link_libraries(sara
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  DO_Sara_VideoIO
)

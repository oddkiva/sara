set(Python_ADDITIONAL_VERSIONS 3.4)
find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python-py34 REQUIRED)


set(VIRTUAL_ENV $ENV{VIRTUAL_ENV})
message("VIRTUAL_ENV ${VIRTUAL_ENV}")


include_directories(
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${VIRTUAL_ENV}/lib/python3.4/site-packages/numpy/core/include
)

message("${VIRTUAL_ENV}/lib/python3.4/site-packages/numpy/core/include")

file(GLOB sara_SRC_FILES FILES *.cpp)
python_add_module(sara ${sara_SRC_FILES})
target_link_libraries(sara
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  DO_Sara_ImageIO
  DO_Sara_VideoIO
)

find_package(
  DO_Sara
  COMPONENTS SfM
  REQUIRED)

macro(sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER
                                          "Examples/Sara/FeatureDescriptors")
endmacro()

sara_add_example(sift_example)
target_link_libraries(sift_example PRIVATE Boost::filesystem)

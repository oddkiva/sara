find_package(DO_Sara COMPONENTS Core Graphics REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "Examples/Sara/Graphics")
endmacro()

file(GLOB example_SOURCE_FILES FILES *.cpp)

foreach (file ${example_SOURCE_FILES})
  get_filename_component(filename "${file}" NAME_WE)
  sara_add_example(${filename})
endforeach ()

find_package(DO_Sara COMPONENTS Core Graphics VideoIO ImageProcessing REQUIRED)

macro (sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES
                        COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE ${DO_Sara_LIBRARIES})
  set_property(TARGET ${example} PROPERTY FOLDER "Examples/Sara/VideoIO")
endmacro ()

sara_add_example(video_read_example)
sara_add_example(video_write_example)

macro(sara_add_example example)
  add_executable(${example} ${example}.cpp)
  set_target_properties(${example} PROPERTIES COMPILE_FLAGS ${SARA_DEFINITIONS})
  target_link_libraries(${example} PRIVATE DO::Sara::Graphics DO::Sara::ImageIO)
  set_property(TARGET ${example} PROPERTY FOLDER "Examples/Sara/ImageIO")
endmacro()

sara_add_example(read_write_heif_images)
sara_add_example(read_write_webp_images)

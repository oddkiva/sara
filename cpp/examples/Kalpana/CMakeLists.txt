add_custom_target(copy_data_folder_to_binary_dir)
add_custom_command(
  TARGET copy_data_folder_to_binary_dir
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/
    # Check append ("Debug", "Release") subfolder if we are in a multi-config
    # project.
    ${CMAKE_BINARY_DIR}/bin/$<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>>data
)
set_property(TARGET copy_data_folder_to_binary_dir #
             PROPERTY FOLDER "Examples")

add_subdirectory(EasyGL)
add_subdirectory(Emscripten)
add_subdirectory(Qt)
add_subdirectory(ImPlot)

add_custom_target(copy_data_folder_to_binary_dir)
if (CMAKE_CONFIGURATION_TYPES)
  add_custom_command(
    TARGET copy_data_folder_to_binary_dir
    POST_BUILD
    COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/
    # Append ("Debug", "Release") subfolder if we are in a multi-config
    # project.
    ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/data
  )
else ()
  add_custom_command(
    TARGET copy_data_folder_to_binary_dir
    POST_BUILD
    COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data/
    ${CMAKE_BINARY_DIR}/bin/data
  )
endif()
set_property(TARGET copy_data_folder_to_binary_dir #
             PROPERTY FOLDER "Examples")

add_subdirectory(EasyGL)
add_subdirectory(Emscripten)
add_subdirectory(Qt)
add_subdirectory(ImPlot)

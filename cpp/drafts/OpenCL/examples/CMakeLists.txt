add_subdirectory(hello_opencl)
add_subdirectory(image_processing)

file(GLOB example_FILES FILES *.cpp)

foreach (file ${example_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  add_executable(${filename} ${file})
  target_compile_definitions(${filename} PRIVATE -DGL_SILENCE_DEPRECATION)
  target_include_directories(${filename} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${filename}
    PRIVATE ${DO_Sara_LIBRARIES}
            ${HDF5_LIBRARIES}
            DO_Kalpana_Math
            DO_Kalpana_3D
            glfw)
  target_link_libraries(${filename} PRIVATE DO_Sara_OpenCL_GL)
  set_target_properties(${filename} PROPERTIES
    COMPILE_FLAGS "-DSRCDIR=${CMAKE_CURRENT_SOURCE_DIR}")

  if (MSVC)
    add_custom_command(TARGET ${filename} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${GLEW_DLL}"
      "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/glew32.dll")
  endif ()
endforeach ()

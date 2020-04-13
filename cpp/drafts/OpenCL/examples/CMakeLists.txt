add_subdirectory(hello_opencl)
add_subdirectory(image_processing)

file(GLOB example_FILES FILES *.cpp)

foreach (file ${example_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  add_executable(${filename} ${file})
  target_include_directories(${filename}
    PRIVATE
    ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${filename}
    PRIVATE
    ${DO_Sara_LIBRARIES}
    DO_Sara_OpenCL_GL
    DO_Kalpana_Math
    ${HDF5_LIBRARIES}
    glfw)
  target_compile_definitions(${filename}
    PRIVATE
    SRCDIR=${CMAKE_CURRENT_SOURCE_DIR})

  if (MSVC)
    add_custom_command(TARGET ${filename} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${GLEW_DLL}"
      "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/glew32.dll")
  endif ()

  set_property(TARGET ${filename} PROPERTY FOLDER "Tutorials/OpenGL")
endforeach ()

file(GLOB example_FILES FILES *.cpp)

foreach(file ${example_FILES})
  get_filename_component(filename "${file}" NAME_WE)

  add_executable(${filename} ${file})
  target_include_directories(${filename} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(
    ${filename}
    PRIVATE DO::Sara::Core #
            DO::Sara::ImageIO #
            DO::Kalpana::EasyGL #
            ${HDF5_LIBRARIES} #
            glfw)
  target_compile_definitions(
    ${filename}
    PRIVATE
      $<$<PLATFORM_ID:Darwin>:GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED>
  )
  target_compile_definitions(
    ${filename} #
    PRIVATE $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:DO_SARA_STATIC> #
  )

  add_dependencies(${filename} copy_data_folder_to_binary_dir)

  set_property(TARGET ${filename} PROPERTY FOLDER "Examples/Kalpana/EasyGL")
endforeach()

target_link_libraries(hello_video PRIVATE DO::Sara::VideoIO)

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
            # ${HDF5_LIBRARIES} #
            glfw)
  target_compile_definitions(
    ${filename}
    PRIVATE SRCDIR=${CMAKE_CURRENT_SOURCE_DIR} #
            $<$<PLATFORM_ID:Darwin>:GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED>)

  if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${filename} PRIVATE DO_SARA_STATIC)
  endif()

  set_property(TARGET ${filename} PROPERTY FOLDER "Examples/Kalpana/EasyGL")
endforeach()

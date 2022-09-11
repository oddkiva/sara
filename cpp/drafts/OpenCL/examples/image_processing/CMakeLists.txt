file(GLOB SRC_FILES FILES *.cpp *.hpp *.cl)
source_group("OpenCL Source Files" REGULAR_EXPRESSION
  "${CMAKE_CURRENT_SOURCE_DIR}/.*\\.cl")

add_executable(image_processing ${SRC_FILES})

target_link_libraries(image_processing
  PRIVATE
  DO::Sara::Core
  DO::Sara::ImageIO
  DO::Sara::ImageProcessing
  DO::Sara::OpenCL
  DO::Kalpana::EasyGL
  glfw)

target_compile_definitions(image_processing
  PRIVATE
  SRCDIR=${CMAKE_CURRENT_SOURCE_DIR}
  $<$<PLATFORM_ID:Darwin>:GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED>)

set_target_properties(image_processing
  PROPERTIES
  FOLDER "Tutorials/OpenCL")

find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

cuda_add_executable(nvidia_decoder_example nvidia_decoder_example.cpp)
target_link_libraries(nvidia_decoder_example
  PRIVATE
  ${FFMPEG_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  glfw
  nvidia::VideoCodecHelpers)
set_target_properties(nvidia_decoder_example
  PROPERTIES
  FOLDER "Examples/DRAFTS")
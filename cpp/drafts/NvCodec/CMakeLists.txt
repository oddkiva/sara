find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(glfw3 REQUIRED)


cuda_add_executable(NvCodecExample AppDecGL.cpp AppDecUtils.h FramePresenterGL.h)
target_include_directories(NvCodecExample
  PRIVATE
  ${GLUT_INCLUDE_DIR})
target_link_libraries(NvCodecExample
  PRIVATE
  ${FFMPEG_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${GLUT_LIBRARIES}
  nvidia::VideoCodecHelpers)
set_target_properties(NvCodecExample
  PROPERTIES
  FOLDER "Examples/DRAFTS")

cuda_add_executable(nvidia_decoder_example nvidia_decoder_example.cpp)
target_link_libraries(nvidia_decoder_example
  PRIVATE
  ${FFMPEG_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${OPENGL_LIBRARIES}
  glfw
  nvidia::VideoCodecHelpers)
set_target_properties(nvidia_decoder_example
  PROPERTIES
  FOLDER "Examples/DRAFTS")
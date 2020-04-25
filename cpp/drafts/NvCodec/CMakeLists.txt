find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)


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

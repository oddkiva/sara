if (NOT APPLE)
  find_package(GLEW REQUIRED)
endif ()

cuda_add_executable(NvCodecExample AppDecGL.cpp AppDecUtils.h FramePresenterGL.h)
target_link_libraries(NvCodecExample
  ${FFMPEG_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  glut
  nvidia::VideoCodecHelpers)

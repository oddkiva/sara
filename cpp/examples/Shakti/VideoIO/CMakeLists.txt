find_package(
  DO_Sara
  COMPONENTS Core Graphics ImageIO
  REQUIRED)
find_package(glfw3 REQUIRED)

cuda_add_executable(nvidia_decoder_example nvidia_decoder_example.cpp)
target_include_directories(
  nvidia_decoder_example PRIVATE ${CMAKE_SOURCE_DIR}/cpp/src
                                 ${CMAKE_SOURCE_DIR}/cpp/third-party)
target_link_libraries(
  nvidia_decoder_example
  PRIVATE DO::Shakti::VideoIO
          ${CUDA_CUDA_LIBRARY}
          ${OPENGL_LIBRARIES}
          ${GLEW_LIBRARIES}
          ${DO_Sara_LIBRARIES}
          glfw)
set_target_properties(
  nvidia_decoder_example
  PROPERTIES
  FOLDER "Examples/Shakti")

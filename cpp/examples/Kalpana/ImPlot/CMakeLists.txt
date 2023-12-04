add_executable(implot_example implot_example.cpp)
target_link_libraries(
  implot_example
  PRIVATE DO::Sara::Core #
          DO::Sara::ImageIO #
          DO::Kalpana::EasyGL #
          ImGui
          ImPlot
          ${HDF5_LIBRARIES} #
          glfw)
target_compile_definitions(
  implot_example
  PRIVATE
    $<$<PLATFORM_ID:Darwin>:GL_DO_NOT_WARN_IF_MULTI_GL_VERSION_HEADERS_INCLUDED>
)
target_compile_definitions(
  implot_example
  PRIVATE $<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:DO_SARA_STATIC> #
)

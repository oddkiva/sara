add_library(
  ImPlot
  implot.h implot.cpp #
  implot_internal.h #
  implot_items.cpp #
  # implot_demo.cpp
)
target_include_directories(ImPlot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ImPlot PUBLIC ImGui)

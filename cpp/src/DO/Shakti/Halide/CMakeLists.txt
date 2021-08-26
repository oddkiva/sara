set(BUILD_SHARED_LIBS OFF)

add_subdirectory(Components)
add_subdirectory(Generators)

add_library(DO_Shakti_Halide_Backend Backend.hpp Backend.cpp)
target_link_libraries(DO_Shakti_Halide_Backend
  PRIVATE
  DO::Sara::Core
  Halide::Runtime
  shakti_gaussian_convolution_cpu
)
set_target_properties(DO_Shakti_Halide_Backend
  PROPERTIES
  FOLDER "Libraries/Shakti/Halide"
)

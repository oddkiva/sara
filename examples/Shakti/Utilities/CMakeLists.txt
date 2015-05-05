file(GLOB CUDA_UTILITIES_FILES FILES *.hpp *.cpp)

if (MSVC)
  add_definitions(-D _CRT_SECURE_NO_WARNINGS)
endif ()
add_library(DO_Shakti_Utilities ${CUDA_UTILITIES_FILES})
target_link_libraries(DO_Shakti_Utilities ${CUDA_CUDART_LIBRARY})
set_property(TARGET DO_Shakti_Utilities PROPERTY FOLDER "Shakti Libraries")

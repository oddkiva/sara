# ==============================================================================
# The region growing algorithm.
set(HyperParameterLearning_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HyperParameterLearning_INC
  DebugEllipseInterArea.hpp)
set(HyperParameterLearning_SRC
  DebugEllipseInterArea.cpp)
set(HyperParameterLearning_DEPS "${DO_Sara_LIBRARIES};DO::Sara::Region;DO::Sara::EmpiricalValidation")

sara_make_library("HyperParameterLearning"
  "${HyperParameterLearning_INC_DIR}"
  "${HyperParameterLearning_INC}"
  "${HyperParameterLearning_SRC}"
  "${HyperParameterLearning_DEPS}")

# ==============================================================================
# The region growing algorithm.
set(HyperParameterLearning_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HyperParameterLearning_INC
  DebugEllipseInterArea.hpp
  Learn_P_d.hpp
  Learn_P_f.hpp)
set(HyperParameterLearning_SRC
  DebugEllipseInterArea.cpp
  Learn_P_d.cpp
  Learn_P_f.cpp)
set(HyperParameterLearning_DEPS
  "${DO_Sara_LIBRARIES};DO::Sara::MatchPropagation;DO::Sara::EmpiricalValidation")

sara_make_library("HyperParameterLearning"
  "${HyperParameterLearning_INC_DIR}"
  "${HyperParameterLearning_INC}"
  "${HyperParameterLearning_SRC}"
  "${HyperParameterLearning_DEPS}")

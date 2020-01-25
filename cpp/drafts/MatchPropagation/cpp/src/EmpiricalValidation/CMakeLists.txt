# ==============================================================================
# Helper classes to perform studies with Mikolajczyk et al.'s dataset.
add_library(DO_MikolajczykDataset
  MikolajczykDataset.cpp MikolajczykDataset.hpp
  StudyOnMikolajczykDataset.cpp StudyOnMikolajczykDataset.hpp)
#do_use_modules(DO_MikolajczykDataset "${DO_Sara_LIBRARIES};DO_Stat")

set_property(TARGET DO_MikolajczykDataset PROPERTY FOLDER "DO 4th-Order Match Propagation")

# ==============================================================================
# Validate empirically all the parameters.
add_library(DO_StudyRegionGrowingParameters
  DebugEllipseInterArea.cpp
  DebugEllipseInterArea.hpp
  Learn_P_f.cpp
  Learn_P_f.hpp
  EvaluateQualityOfLocalAffineApproximation.cpp
  EvaluateQualityOfLocalAffineApproximation.hpp
  EvaluateOutlierResistance.cpp
  EvaluateOutlierResistance.hpp
  Study_N_K_m.cpp
  Study_N_K_m.hpp
  StudyPerfWithHatN_K.cpp
  StudyPerfWithHatN_K.hpp)
do_use_modules(DO_StudyRegionGrowingParameters 
  "${DO_Sara_LIBRARIES};DO_MikolajczykDataset;DO_GrowRegion;DO_Stat")
if (OPENMP_FOUND AND NOT WIN32 AND NOT APPLE)
  target_link_libraries(DO_StudyRegionGrowingParameters gomp)
endif()
set_property(TARGET DO_StudyRegionGrowingParameters PROPERTY FOLDER "DO 4th-Order Match Propagation")

add_executable(DO_ValidateRegionGrowingNeighborhoodParams
  DO_ValidateRegionGrowingNeighborhoodParams.cpp)
do_use_modules(DO_ValidateRegionGrowingNeighborhoodParams
  "${DO_Sara_LIBRARIES};DO_StudyRegionGrowingParameters")
if (OPENMP_FOUND AND NOT WIN32 AND NOT APPLE)
  target_link_libraries(DO_ValidateRegionGrowingNeighborhoodParams gomp)
endif()
set_property(TARGET DO_ValidateRegionGrowingNeighborhoodParams PROPERTY FOLDER "DO 4th-Order Match Propagation")

# ==============================================================================
# For performance measurement and empirical validation.
add_library(DO_Stat Stat.cpp Stat.hpp)
set_property(TARGET DO_Stat PROPERTY FOLDER "DO 4th-Order Match Propagation")

# ==============================================================================
# Region data structures
add_library(DO_Region
            Region.cpp Region.hpp
            RegionBoundary.cpp RegionBoundary.hpp)
if (DEFINED ENABLE_CXX11)
  set_target_properties(DO_Region PROPERTIES COMPILE_FLAGS ${ENABLE_CXX11})
endif ()
do_use_modules(DO_Region ${DO_LIBRARIES})
set_property(TARGET DO_Region PROPERTY FOLDER "DO 4th-Order Match Propagation")

# ==============================================================================
# The region growing algorithm.
add_library(DO_GrowRegion
            LocalAffineConsistency.cpp LocalAffineConsistency.hpp
            MatchNeighborhood.cpp MatchNeighborhood.hpp
            GrowthParams.hpp
            DynamicMatchGraph.hpp
            GrowRegionDebugMacros.hpp
            GrowRegion.cpp GrowRegion.hpp
            GrowMultipleRegions.cpp GrowMultipleRegions.hpp
            RegionGrowingAnalyzer.cpp RegionGrowingAnalyzer.hpp)
if (DEFINED ENABLE_CXX11)
  set_target_properties(DO_GrowRegion PROPERTIES COMPILE_FLAGS ${ENABLE_CXX11})
endif ()
do_use_modules(DO_GrowRegion "${DO_LIBRARIES};DO_Region;DO_Stat")
set_property(TARGET DO_GrowRegion PROPERTY FOLDER "DO 4th-Order Match Propagation")


# ==============================================================================
# Helper classes to perform studies with Mikolajczyk et al.'s dataset.
add_library(DO_MikolajczykDataset
            MikolajczykDataset.cpp MikolajczykDataset.hpp
            StudyOnMikolajczykDataset.cpp StudyOnMikolajczykDataset.hpp)
do_use_modules(DO_MikolajczykDataset "${DO_LIBRARIES};DO_Stat")
if (DEFINED ENABLE_CXX11)
  set_target_properties(DO_GrowRegion PROPERTIES COMPILE_FLAGS ${ENABLE_CXX11})
endif ()

set_property(TARGET DO_MikolajczykDataset PROPERTY FOLDER "DO 4th-Order Match Propagation")


if (DO_BUILD_EXPERIMENTAL_VALIDATION_PROGRAMS)
  # ==============================================================================
  # Define the SRCPATH macro for the DO.Core library.
  add_definitions(-DVM_DATA_DIR=${VM_DATA_DIR})

  add_executable(DO_MikolajczykDataset_test DO_MikolajczykDataset_test.cpp)
  do_use_modules(DO_MikolajczykDataset_test "${DO_LIBRARIES};DO_MikolajczykDataset")
  set_property(TARGET DO_MikolajczykDataset_test PROPERTY FOLDER "DO 4th-Order Match Propagation")

  add_executable(DO_Region_test DO_Region_test.cpp)
  do_use_modules(DO_Region_test "${DO_LIBRARIES};DO_MikolajczykDataset;DO_Region")
  set_property(TARGET DO_Region_test PROPERTY FOLDER "DO 4th-Order Match Propagation")

  add_executable(DO_GrowRegion_test DO_GrowRegion_test.cpp)
  do_use_modules(DO_GrowRegion_test
                 "${DO_LIBRARIES};DO_GrowRegion;DO_MikolajczykDataset;DO_Region;DO_LoweSIFTWrapper")
  set_property(TARGET DO_GrowRegion_test PROPERTY FOLDER "DO 4th-Order Match Propagation")
                 
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
                 "${DO_LIBRARIES};DO_MikolajczykDataset;DO_GrowRegion;DO_Stat")
  if (OPENMP_FOUND AND NOT WIN32 AND NOT APPLE)
    target_link_libraries(DO_StudyRegionGrowingParameters gomp)
  endif()
  set_property(TARGET DO_StudyRegionGrowingParameters PROPERTY FOLDER "DO 4th-Order Match Propagation")

  add_executable(DO_ValidateRegionGrowingNeighborhoodParams
                 DO_ValidateRegionGrowingNeighborhoodParams.cpp)
  do_use_modules(DO_ValidateRegionGrowingNeighborhoodParams
                 "${DO_LIBRARIES};DO_StudyRegionGrowingParameters")
  if (OPENMP_FOUND AND NOT WIN32 AND NOT APPLE)
    target_link_libraries(DO_ValidateRegionGrowingNeighborhoodParams gomp)
  endif()
  set_property(TARGET DO_ValidateRegionGrowingNeighborhoodParams PROPERTY FOLDER "DO 4th-Order Match Propagation")
endif ()
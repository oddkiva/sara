find_package(DO_Sara Graphics REQUIRED)


# ==============================================================================
# Region data structures
add_library(DO_Region
            Region.cpp Region.hpp
            RegionBoundary.cpp RegionBoundary.hpp)
# do_use_modules(DO_Region ${DO_Sara_LIBRARIES})
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
# do_use_modules(DO_GrowRegion "${DO_Sara_LIBRARIES};DO_Region;DO_Stat")
set_property(TARGET DO_GrowRegion PROPERTY FOLDER "DO 4th-Order Match Propagation")


add_subdirectory(HyperParameterLearning)

if (SARA_BUILD_EXPERIMENTAL_VALIDATION_PROGRAMS)
  add_subdirectory(EmpiricalValidation)
endif ()

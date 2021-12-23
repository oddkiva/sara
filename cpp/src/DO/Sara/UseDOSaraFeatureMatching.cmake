if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureMatching_ADDED GLOBAL PROPERTY
               _DO_Sara_FeatureMatching_INCLUDED)
  if (NOT DO_Sara_FeatureMatching_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureMatching)
    sara_create_common_variables("FeatureMatching")
    sara_generate_library("FeatureMatching")
    target_link_libraries(DO_Sara_FeatureMatching
      PRIVATE
      DO::Sara::Match
      DO::Sara::KDTree
      PUBLIC
      DO::Sara::Core
      DO::Sara::Features
      $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  endif ()
endif ()

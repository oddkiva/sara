if (SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_FeatureDetectors_ADDED GLOBAL PROPERTY _DO_Sara_FeatureDetectors_INCLUDED)
  if (NOT DO_Sara_FeatureDetectors_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/FeatureDetectors)
    sara_create_common_variables("FeatureDetectors")
    sara_generate_library("FeatureDetectors")

    if(SARA_USE_HALIDE)
      target_compile_definitions(DO_Sara_ImageProcessing
                                 PRIVATE DO_SARA_USE_HALIDE)
    endif ()

    target_link_libraries(DO_Sara_FeatureDetectors
      PRIVATE
      DO::Sara::Geometry
      DO::Sara::FeatureDescriptors
      DO::Sara::ImageProcessing
      PUBLIC
      DO::Sara::Core
      $<$<BOOL:OpenMP_CXX_FOUND>:OpenMP::OpenMP_CXX>)
  endif ()
endif ()

if(SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageProcessing_ADDED GLOBAL
               PROPERTY _DO_Sara_ImageProcessing_INCLUDED)
  if(NOT DO_Sara_ImageProcessing_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/ImageProcessing)
    sara_create_common_variables("ImageProcessing")
    sara_generate_library("ImageProcessing")
    target_link_libraries(
      DO_Sara_ImageProcessing
      PRIVATE Halide::Halide
              Halide::Runtime
              shakti_subtract_32f_cpu
              shakti_gaussian_convolution_cpu
              shakti_gaussian_convolution_gpu)
  endif()
endif()

if(SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageProcessing_ADDED GLOBAL
               PROPERTY _DO_Sara_ImageProcessing_INCLUDED)
  if(NOT DO_Sara_ImageProcessing_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/ImageProcessing)
    sara_create_common_variables("ImageProcessing")
    sara_generate_library("ImageProcessing")

    if(SARA_USE_HALIDE)
      target_compile_definitions(DO_Sara_ImageProcessing
                                 PRIVATE DO_SARA_USE_HALIDE)
      target_link_libraries(
        DO_Sara_ImageProcessing
        PUBLIC ${CMAKE_DL_LIBS}
        PRIVATE DO::Sara::Core
                Halide::Halide
                Halide::Runtime
                # Fast color conversion
                shakti_rgb8u_to_gray32f_cpu
                # Binary operations.
                shakti_subtract_32f_cpu
                # Resize operations.
                shakti_scale_32f_cpu
                shakti_reduce_32f_cpu
                shakti_enlarge_cpu
                # Differential operations.
                shakti_gradient_2d_32f_cpu
                shakti_polar_gradient_2d_32f_cpu
                # Gaussian convolutions.
                shakti_gaussian_convolution_cpu)
    endif()
  endif()
endif()

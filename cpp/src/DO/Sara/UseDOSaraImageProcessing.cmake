if(SARA_USE_FROM_SOURCE)
  get_property(DO_Sara_ImageProcessing_ADDED GLOBAL
               PROPERTY _DO_Sara_ImageProcessing_INCLUDED)
  if(NOT DO_Sara_ImageProcessing_ADDED)
    sara_glob_directory(${DO_Sara_SOURCE_DIR}/ImageProcessing)
    sara_create_common_variables("ImageProcessing")
    sara_generate_library("ImageProcessing")

    target_link_libraries(DO_Sara_ImageProcessing PRIVATE DO::Sara::Core)

    if(SARA_USE_HALIDE)
      target_compile_definitions(DO_Sara_ImageProcessing
                                 PRIVATE DO_SARA_USE_HALIDE)
      target_link_libraries(
        DO_Sara_ImageProcessing
        PUBLIC ${CMAKE_DL_LIBS}
        PRIVATE Halide::Halide
                # Fast color conversion
                shakti_rgb8u_to_gray32f_cpu
                shakti_bgra8u_to_gray32f_cpu
                # Binary operations.
                shakti_subtract_32f_cpu
                # Cartesian to polar coordinates.
                shakti_cartesian_to_polar_32f_cpu
                # Resize operations.
                shakti_scale_32f_cpu
                shakti_reduce_32f_cpu
                shakti_enlarge_cpu
                # Rotate functions.
                shakti_rotate_cw_90_rgb8_cpu
                # Differential operations.
                shakti_gradient_2d_32f_cpu
                shakti_polar_gradient_2d_32f_cpu
                # Separable kernel convolution 2D
                shakti_separable_convolution_2d_cpu
                # Gaussian convolutions.
                shakti_gaussian_convolution_cpu
                # Moment matrix
                shakti_moment_matrix_32f_cpu
                # Cornerness
                shakti_cornerness_32f_cpu
                # Local extremum map
                shakti_scale_space_dog_extremum_32f_cpu
                $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>)
    endif()
  endif()
endif()

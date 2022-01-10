add_library(
  DO_Shakti_Cuda_FeatureDetectors
  SIFT/Octave.hpp
  SIFT/SiftParameters.hpp
  TunedConvolutions/GaussianOctaveKernels.hpp
  TunedConvolutions/SmallGaussianConvolutionFP16.hpp
  TunedConvolutions/SmallGaussianConvolutionFP32.hpp
  TunedConvolutions/SmallGaussianConvolutionFP32.cu)
target_compile_options(DO_Shakti_Cuda_FeatureDetectors
  PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(DO_Shakti_Cuda_FeatureDetectors PRIVATE DO::Sara::Core)
add_library(DO::Shakti::Cuda::FeatureDetectors ALIAS
            DO_Shakti_Cuda_FeatureDetectors)

add_subdirectory(test)

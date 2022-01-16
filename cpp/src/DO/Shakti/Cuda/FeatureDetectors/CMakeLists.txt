add_library(DO_Shakti_Cuda_FeatureDetectors
  Octave.hpp
  SIFT/SiftParameters.hpp
  TunedConvolutions/GaussianOctaveKernels.hpp
  TunedConvolutions/SmallGaussianConvolutionFP16.hpp
  TunedConvolutions/SmallGaussianConvolutionFP32.hpp
  TunedConvolutions/SmallGaussianConvolutionFP32.cu)

set_target_properties(DO_Shakti_Cuda_FeatureDetectors
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE ON
  FOLDER "Libraries/Shakti/Cuda")

target_compile_options(DO_Shakti_Cuda_FeatureDetectors
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

target_link_libraries(DO_Shakti_Cuda_FeatureDetectors
  PRIVATE
  DO::Sara::Core DO::Shakti::Cuda::Utilities)

add_library(DO::Shakti::Cuda::FeatureDetectors ALIAS
            DO_Shakti_Cuda_FeatureDetectors)

add_subdirectory(examples)
add_subdirectory(test)

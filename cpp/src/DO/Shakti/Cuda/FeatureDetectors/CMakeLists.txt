add_library(
  DO_Shakti_Cuda_FeatureDetectors
  Octave.hpp
  SIFT/SiftParameters.hpp
  TunedConvolutions/GaussianOctaveKernels.hpp
  TunedConvolutions/SmallGaussianConvolutionFP16.hpp
  TunedConvolutions/SmallGaussianConvolutionFP32.hpp
  TunedConvolutions/SmallGaussianConvolutionFP32.cu
  TunedConvolutions/GaussianOctaveComputer.hpp
  TunedConvolutions/GaussianOctaveComputer.cu
  DoG.hpp
  DoG.cu
  Gradient.hpp
  Gradient.cu
  DominantOrientations.hpp
  DominantOrientations.cu
  ScaleSpaceExtremum.hpp
  ScaleSpaceExtremum.cu)

set_target_properties(
  DO_Shakti_Cuda_FeatureDetectors
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED YES
             POSITION_INDEPENDENT_CODE ON
             FOLDER "Libraries/Shakti/Cuda")

set(compile_options --expt-relaxed-constexpr --extended-lambda)
target_compile_options(
  DO_Shakti_Cuda_FeatureDetectors
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${compile_options}>)

target_link_libraries(DO_Shakti_Cuda_FeatureDetectors
                      PRIVATE DO::Sara::Core DO::Shakti::Cuda::Utilities)

add_library(DO::Shakti::Cuda::FeatureDetectors ALIAS
            DO_Shakti_Cuda_FeatureDetectors)

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>


namespace DO { namespace Sara {

  Set<OERegion, RealDescriptor> compute_sift_keypoints(const Image<float>& image);

} /* namespace Sara */
} /* namespace DO */

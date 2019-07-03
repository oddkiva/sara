#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/FeatureDescriptors.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureMatching.hpp>


namespace DO::Sara {

auto compute_sift_keypoints(const Image<float>& image)
    -> Set<OERegion, RealDescriptor>;

} /* namespace DO::Sara */

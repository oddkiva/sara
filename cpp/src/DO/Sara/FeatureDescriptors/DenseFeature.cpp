#include <DO/Sara/FeatureDescriptors.hpp>


namespace DO { namespace Sara {

  Image<ComputeSIFTDescriptor<>::descriptor_type>
  compute_dense_sift(const ImageView<float>& image, int local_patch_size)
  {
    auto compute_dense_feature = DenseFeatureComputer<ComputeSIFTDescriptor<>>{};
    return compute_dense_feature(image, local_patch_size);
  }

} /* namespace Sara */
} /* namespace DO */

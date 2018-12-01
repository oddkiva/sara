#include "sift.hpp"

#include <DO/Sara/Core.hpp>


namespace DO { namespace Sara {

  Set<OERegion, RealDescriptor>
  compute_sift_keypoints(const Image<float>& image)
  {
    using namespace std;

    // Time everything.
    auto timer = Timer{};
    auto elapsed = 0.;

    // We describe the work flow of the feature detection and description.
    auto keys = Set<OERegion, RealDescriptor>{};
    auto& DoGs = keys.features;
    auto& SIFTDescriptors = keys.descriptors;

    // 1. Feature extraction.
    print_stage("Computing DoG extrema");
    timer.restart();
    ImagePyramidParams pyr_params(0);
    ComputeDoGExtrema compute_DoGs{pyr_params, 0.01f};
    auto scale_octave_pairs = vector<Point2i>{};
    DoGs = compute_DoGs(image, &scale_octave_pairs);
    auto dog_detection_time = timer.elapsed_ms();
    elapsed += dog_detection_time;
    cout << "DoG detection time = " << dog_detection_time << " ms" << endl;
    cout << "DoGs.size() = " << DoGs.size() << endl;

    // 2. Feature orientation.
    // Prepare the computation of gradients on gaussians.
    print_stage("Computing gradients of Gaussians");
    timer.restart();
    auto nabla_G = gradient_polar_coordinates(compute_DoGs.gaussians());
    auto grad_gaussian_time = timer.elapsed_ms();
    elapsed += grad_gaussian_time;
    cout << "gradient of Gaussian computation time = " << grad_gaussian_time
         << " ms" << endl;
    cout << "DoGs.size() = " << DoGs.size() << endl;


    // Find dominant gradient orientations.
    print_stage(
        "Assigning (possibly multiple) dominant orientations to DoG extrema");
    timer.restart();
    ComputeDominantOrientations assign_dominant_orientations;
    assign_dominant_orientations(nabla_G, DoGs, scale_octave_pairs);
    auto ori_assign_time = timer.elapsed_ms();
    elapsed += ori_assign_time;
    cout << "orientation assignment time = " << ori_assign_time << " ms"
         << endl;
    cout << "DoGs.size() = " << DoGs.size() << endl;


    // 3. Feature description.
    print_stage("Describe DoG extrema with SIFT descriptors");
    timer.restart();
    ComputeSIFTDescriptor<> compute_sift;
    SIFTDescriptors = compute_sift(DoGs, scale_octave_pairs, nabla_G);
    auto sift_description_time = timer.elapsed_ms();
    elapsed += sift_description_time;
    cout << "description time = " << sift_description_time << " ms" << endl;
    cout << "sifts.size() = " << SIFTDescriptors.size() << endl;

    // Summary in terms of computation time.
    print_stage("Total Detection/Description time");
    cout << "SIFT computation time = " << elapsed << " ms" << endl;

    // 4. Rescale  the feature position and scale $(x,y,\sigma)$ with the octave
    //    scale.
    for (size_t i = 0; i != DoGs.size(); ++i)
    {
      auto octave_scale_factor =
          nabla_G.octave_scaling_factor(scale_octave_pairs[i](1));
      DoGs[i].center() *= octave_scale_factor;
      DoGs[i].shape_matrix() /= pow(octave_scale_factor, 2);
    }

    return keys;
  }

} /* namespace Sara */
} /* namespace DO */

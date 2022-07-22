// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Core/StdVectorHelpers.hpp>
#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/ImageProcessing/Cornerness.hpp>


using namespace std;


namespace DO { namespace Sara {

  auto harris_cornerness(const ImageView<float>& gx, const ImageView<float>& gy,
                         float sigma_I, float kappa) -> Image<float>
  {
    auto m = Tensor_<float, 3>{3, gx.height(), gx.width()};
    auto mxx = image_view(m[0]);
    auto myy = image_view(m[1]);
    auto mxy = image_view(m[2]);
    second_moment_matrix(gx, gy, mxx, myy, mxy);

    auto m_sigma_I = Tensor_<float, 3>{m.sizes()};
    for (auto i = 0; i < 3; ++i)
      image_view(m_sigma_I[i]) = image_view(m[i]).compute<Gaussian>(sigma_I);

    auto cornerness = Image<float>{gx.sizes()};
    compute_cornerness(image_view(m_sigma_I[0]),  //
                       image_view(m_sigma_I[1]),  //
                       image_view(m_sigma_I[2]),  //
                       kappa,                     //
                       cornerness);

    return cornerness;
  }

  auto scale_adapted_harris_cornerness(const ImageView<float>& I, float sigma_I,
                                       float sigma_D, float kappa)
      -> Image<float>
  {
#ifdef DO_SARA_USE_HALIDE
    const auto I_sigma_D = I.compute<Gaussian>(sigma_D);

    auto g = Tensor_<float, 3>{2, I.height(), I.width()};
    auto gx = image_view(g[0]);
    auto gy = image_view(g[1]);
    gradient(I_sigma_D, gx, gy);

    const auto m = second_moment_matrix(g);

    auto m_sigma_I = Tensor_<float, 3>{m.sizes()};
    for (auto i = 0; i < 3; ++i)
      image_view(m_sigma_I[i]) = image_view(m[i]).compute<Gaussian>(sigma_I);

    auto cornerness = Image<float>{I.sizes()};
    compute_cornerness(image_view(m_sigma_I[0]),  //
                       image_view(m_sigma_I[1]),  //
                       image_view(m_sigma_I[2]),  //
                       kappa,                     //
                       cornerness);
#else
    const auto M = I  //
                       .compute<Gaussian>(sigma_D)
                       .compute<Gradient>()
                       .compute<SecondMomentMatrix>()
                       .compute<Gaussian>(sigma_I);

    auto cornerness = Image<float>{I.sizes()};
    std::transform(M.begin(), M.end(), cornerness.begin(),
                   [kappa](const auto& m) {
                     return m.determinant() - kappa * pow(m.trace(), 2);
                   });
#endif

    // Rescale the cornerness function.
    cornerness.flat_array() *= sigma_D * sigma_D;

    return cornerness;
  }

  ImagePyramid<float>
  harris_cornerness_pyramid(const ImageView<float>& image, float kappa,
                            const ImagePyramidParams& params)
  {
    // Resize the image with the appropriate factor.
    const auto resize_factor =
        std::pow(2.f, static_cast<float>(-params.first_octave_index()));
    auto I = enlarge(image, resize_factor);

    // Deduce the new camera sigma with respect to the dilated image.
    const auto camera_sigma =
        static_cast<float>(params.scale_camera()) * resize_factor;

    // Blur the image so that its new sigma is equal to the initial sigma.
    const auto scale_initial = static_cast<float>(params.scale_initial());
    if (camera_sigma < scale_initial)
    {
      const auto sigma = sqrt(square(scale_initial) - square(camera_sigma));
      I = gaussian(I, sigma);
    }

    // Deduce the maximum number of octaves.
    const auto l = std::min(image.width(), image.height());
    const auto b = params.image_padding_size();
    // l/2^k > 2b
    // 2^k < l/(2b)
    // k < log(l/(2b))/log(2)
    const auto octave_count = static_cast<int>(log(l / (2.f * b)) / log(2.f));

    // Shorten names.
    const auto scale_count_per_octave = params.scale_count_per_octave();
    const auto k = float(params.scale_geometric_factor());

    // Create the image pyramid
    auto cornerness = ImagePyramid<float>{};
    cornerness.reset(octave_count, scale_count_per_octave, scale_initial, k);
    for (auto o = 0; o < octave_count; ++o)
    {
      // Compute the octave scaling factor
      cornerness.octave_scaling_factor(o) =
          (o == 0) ? 1.f / resize_factor
                   : cornerness.octave_scaling_factor(o - 1) * 2;

      // Compute the gaussians in octave $o$
      if (o != 0)
        I = downscale(I, 2);

      for (auto s = 0; s < scale_count_per_octave; ++s)
      {
        const auto sigma_I =
            static_cast<float>(cornerness.scale_relative_to_octave(s));
        const auto sigma_D = sigma_I / sqrt(2.f);
        cornerness(s, o) =
            scale_adapted_harris_cornerness(I, sigma_I, sigma_D, kappa);
      }
    }

    return cornerness;
  }

  bool local_min_x(int x, int y, ImageView<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x, y) > I(x + u, y))
        return false;
    return true;
  }

  bool local_min_y(int x, int y, ImageView<float>& I)
  {
    for (int u = -1; u <= 1; ++u)
      if (I(x, y) < I(x + u, y))
        return false;
    return true;
  }

  vector<OERegion>
  ComputeHarrisLaplaceCorners::operator()(const ImageView<float>& I,
                                          vector<Point2i>* scale_octave_pairs)
  {
    auto& G = _gaussians;
    auto& cornerness = _harris;

    G = Sara::gaussian_pyramid(I, _pyr_params);
    cornerness = G;
    for (auto o = 0; o < cornerness.octave_count(); ++o)
    {
      for (auto s = 0; s < cornerness.scale_count_per_octave(); ++s)
      {
        static const auto scale_factor = 1 / std::sqrt(2.f);
        const auto& sigma_I = G.scale_relative_to_octave(s);
        const auto& sigma_D = sigma_I * scale_factor;

        auto M = G(s, o)
                     .compute<Gradient>()
                     .compute<SecondMomentMatrix>()
                     .compute<Gaussian>(sigma_I);

        std::transform(M.begin(), M.end(), cornerness(s, o).begin(),
                       [this](const auto& m) {
                         return m.determinant() - _kappa * pow(m.trace(), 2);
                       });

        // Rescale the cornerness function.
        cornerness(s, o).flat_array() *= sigma_D * sigma_D;
      }
    }

    static constexpr auto preallocated_size = static_cast<int>(1e4);
    auto corners = vector<OERegion>{};
    corners.reserve(preallocated_size);
    if (scale_octave_pairs)
    {
      scale_octave_pairs->clear();
      scale_octave_pairs->reserve(preallocated_size);
    }

    for (auto o = 0; o < cornerness.octave_count(); ++o)
    {
      // Be careful of the bounds. We go from 1 to N-1.
      for (auto s = 1; s < cornerness.scale_count_per_octave(); ++s)
      {
        auto new_corners = laplace_maxima(cornerness, G, s, o, _extremum_thres,
                                          _img_padding_sz, _scale_count,
                                          _extremum_refinement_iter);

        append(corners, new_corners);

        if (scale_octave_pairs)
        {
          for (size_t i = 0; i != new_corners.size(); ++i)
            scale_octave_pairs->push_back(Point2i(s, o));
        }
      }
    }
    shrink_to_fit(corners);
    return corners;
  }

}}  // namespace DO::Sara

// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @example

#ifdef _OPENMP
#include <omp.h>
#endif

#include <set>

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


template <int PatchRadius = 3>
struct LukasKanadeOpticalFlowEstimator
{
  static constexpr auto patch_radius = PatchRadius;
  static constexpr auto patch_size = 2 * PatchRadius + 1;
  static constexpr auto N = patch_size * patch_size;

  auto update_image(const sara::ImageView<float, 2>& I,
                    const sara::ImageView<float, 2>& Ix,
                    const sara::ImageView<float, 2>& Iy) -> void
  {
    _I0.swap(_I1);

    _I1 = I;

    _grad_I = {Ix, Iy};

    if (_dI_dt.sizes() != _I1.sizes())
      _dI_dt.resize(_I1.sizes());
    if (!_I0.empty())
      _dI_dt.matrix() = _I1.matrix() - _I0.matrix();
  }

  auto estimate_flow(const Eigen::Vector2i& p) const -> Eigen::Vector2f
  {
    auto A = Eigen::Matrix<float, N, 2>{};
    auto b = Eigen::Matrix<float, N, 1>{};

    constexpr auto& r = patch_radius;
    auto i = 0;
    for (auto y = p.y() - r; y <= p.y() + r; ++y)
    {
      for (auto x = p.x() - r; x <= p.x() + r; ++x)
      {
        A(i, 0) = _grad_I[0](x, y);
        A(i, 1) = _grad_I[1](x, y);

        b(i) = -_dI_dt(x, y);

        ++i;
      }
    }

    // Check the conditions for solvability.
    const Eigen::Matrix2f AtA = A.transpose() * A;
    const Eigen::Vector2f Atb = A.transpose() * b;

    static constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
    if (std::abs(AtA.determinant()) < 1e-6f)
      return Eigen::Vector2f{nan, nan};

    const Eigen::Vector2f d = AtA.lu().solve(Atb);

    return d;
  }

  auto estimate_flow(const std::vector<Eigen::Vector2i>& ps) const
      -> std::vector<Eigen::Vector2f>
  {
    auto flows = std::vector<Eigen::Vector2f>(ps.size());
    const auto num_points = static_cast<int>(ps.size());
#pragma omp parallel for
    for (auto i = 0; i < num_points; ++i)
      flows[i] = estimate_flow(ps[i]);
    return flows;
  }

  sara::Image<float> _I1;
  sara::Image<float> _I0;

  std::array<sara::Image<float>, 2> _grad_I;
  sara::Image<float> _dI_dt;
};


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

#ifdef _OPENMP
  omp_set_num_threads(omp_get_max_threads());
#endif

  // Input video.
  const auto video_filepath =
      argc < 2 ? "/Users/david/Desktop/Datasets/videos/sample10.mp4" : argv[1];
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();

  // Preprocessing parameters.
  const auto downscale_factor = argc < 3 ? 1 : std::stoi(argv[2]);
  const auto cornerness_thres = argc < 4 ? 0 : std::stof(argv[3]);

  // Harris cornerness parameters.
  //
  // Blur parameter before gradient calculation.
  static const auto sigma_D = std::sqrt(sara::square(1.6f) - 1);
  // Integration domain of the second moment.
  static const auto sigma_I = 2 * sigma_D;
  // Harris cornerness free parameter.
  static constexpr auto kappa = 0.04f;

  // Lukas-Kanade optical flow parameters.
  auto flow_estimator = LukasKanadeOpticalFlowEstimator<>{};

  sara::create_window(video_stream.sizes());
  sara::set_antialiasing();

  auto frames_read = -1;
  constexpr auto skip = 0;
  while (true)
  {
    if (!video_stream.read())
    {
      std::cout << "Reached the end of the video!" << std::endl;
      break;
    }
    ++frames_read;
    if (frames_read % (skip + 1) != 0)
      continue;
    SARA_CHECK(frames_read);

    // Convert to grayscale, downsample.
    sara::tic();
    auto frame_gray = sara::Image<float>{frame.sizes()};
    sara::from_rgb8_to_gray32f(frame, frame_gray);
    sara::toc("Color conversion");

    if (downscale_factor > 1)
    {
      sara::tic();
      frame_gray = frame_gray.compute<sara::Gaussian>(0.5f);
      frame_gray = sara::downscale(frame_gray, downscale_factor);
      sara::toc("Downscale");
    }

    // Calculate the image gradients.
    sara::tic();
    auto frame_gray_blurred = frame_gray.compute<sara::Gaussian>(sigma_D);
    auto gradient = std::array<sara::Image<float>, 2>{};
    std::for_each(gradient.begin(), gradient.end(),
                  [&frame_gray](auto& g) { g.resize(frame_gray.sizes()); });
    sara::gradient(frame_gray_blurred, gradient[0], gradient[1]);
    sara::toc("Gradient");


    // Calculate Harris cornerness functions.
    sara::tic();
    const auto cornerness = sara::harris_cornerness(  //
        gradient[0], gradient[1],                     //
        sigma_I, kappa);
    sara::toc("Harris Cornerness");

    // Select the local maxima of the cornerness functions.
    sara::tic();
    static constexpr auto select = [](const sara::ImageView<float>& cornerness,
                                      const float cornerness_thres) {
      static constexpr auto r = LukasKanadeOpticalFlowEstimator<>::patch_radius;

      const auto extrema = sara::local_maxima(cornerness);

      auto extrema_filtered = std::vector<sara::Point2i>{};
      extrema_filtered.reserve(extrema.size());
      for (const auto& p : extrema)
      {
        const auto in_image_domain =
            (r <= p.x() && p.x() < cornerness.width() - r) &&
            (r <= p.y() && p.y() < cornerness.height() - r);

        if (!in_image_domain)
          continue;

        if (cornerness(p) > cornerness_thres)
          extrema_filtered.emplace_back(p);
      }
      return extrema_filtered;
    };
    const auto corners = select(cornerness, cornerness_thres);
    sara::toc("Corner selection");

    // Calculate the optical flow.
    sara::tic();
    flow_estimator.update_image(frame_gray_blurred, gradient[0], gradient[1]);
    auto flow_vectors = std::vector<Eigen::Vector2f>{};
    if (!flow_estimator._I0.empty())
      flow_vectors = flow_estimator.estimate_flow(corners);
    sara::toc("Optical Flow");

    // TODO: track the corners with optical flow.

    // Draw the sparse flow field.
    if (!flow_estimator._I0.empty())
    {
      const auto num_corners = static_cast<int>(corners.size());
#pragma omp parallel for
      for (auto i = 0; i < num_corners; ++i)
      {
        const auto& v = flow_vectors[i];
        if (std::isnan(v.x()) || std::isnan(v.y()))
          continue;

        const Eigen::Vector2f pa = corners[i].cast<float>() * downscale_factor;
        const Eigen::Vector2f pb = pa + flow_vectors[i] * 10;

// #define SHOW_FLOW
#ifdef SHOW_FLOW
        const auto Y = std::clamp(flow_vectors[i].norm() / 5.f, 0.f, 1.f);
        const auto U = std::clamp(flow_vectors[i].x() / 5, -1.f, 1.f) * 0.5f;
        const auto V = std::clamp(flow_vectors[i].y() / 5, -1.f, 1.f) * 0.5f;
        const auto yuv = sara::Yuv64f(Y, U, V);

        auto rgb32f = sara::Rgb32f{};
        sara::smart_convert_color(yuv, rgb32f);

        auto rgb8 = sara::Rgb8{};
        sara::smart_convert_color(rgb32f, rgb8);
#else
        const auto rgb8 = sara::Red8;
#endif

        constexpr auto& r = LukasKanadeOpticalFlowEstimator<>::patch_radius;
        const auto& l = (2 * r + 1) * downscale_factor;
        const Eigen::Vector2i tl =
            (pa - Eigen::Vector2f::Ones() * r * downscale_factor).cast<int>();

        sara::draw_rect(frame, tl.x(), tl.y(), l, l, rgb8);
        sara::draw_arrow(frame, pa, pb, rgb8, 2);
        sara::fill_circle(frame, int(pa.x() + 0.5f), int(pa.y() + 0.5f), 2,
                          sara::Green8);
      }
      sara::display(frame);
    }
  }

  return 0;
}


int main(int argc, char** argv)
{
  sara::GraphicsApplication app{argc, argv};
  app.register_user_main(__main);
  return app.exec();
}

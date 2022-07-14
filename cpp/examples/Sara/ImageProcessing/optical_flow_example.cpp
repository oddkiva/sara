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

#include <set>

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>


namespace sara = DO::Sara;


template <int PatchRadius = 3>
struct LukasKanadeOpticalFlowEstimator
{
  static constexpr auto patch_radius = PatchRadius;
  static constexpr auto patch_size = 2 * PatchRadius + 1;
  static constexpr auto N = patch_size * patch_size;

  auto update_image(const sara::ImageView<float, 2>& I) -> void
  {
    _I0.swap(_I1);

    _I1 = I;
    if (_blur)
      _I1 = sara::gaussian(_I1, 0.5f);

    _grad_I = sara::gradient(_I1);

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
        A(i, 0) = _grad_I(x, y).x();
        A(i, 1) = _grad_I(x, y).y();

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
    for (auto i = 0u; i < ps.size(); ++i)
      flows[i] = estimate_flow(ps[i]);
    return flows;
  }

  bool _blur = true;

  sara::Image<float> _I1;
  sara::Image<float> _I0;

  sara::Image<Eigen::Vector2f> _grad_I;
  sara::Image<float> _dI_dt;
};


auto localize_zero_crossings(const Eigen::ArrayXf& profile, int num_bins)
    -> std::vector<float>
{
  auto zero_crossings = std::vector<float>{};
  for (auto n = Eigen::Index{}; n < profile.size(); ++n)
  {
    const auto ia = n;
    const auto ib = (n + Eigen::Index{1}) % profile.size();

    const auto& a = profile[ia];
    const auto& b = profile[ib];

    static constexpr auto pi = static_cast<float>(M_PI);
    const auto angle_a = ia * 2.f * M_PI / num_bins;
    const auto angle_b = ib * 2.f * M_PI / num_bins;

    const auto ea = Eigen::Vector2d{std::cos(angle_a),  //
                                    std::sin(angle_a)};
    const auto eb = Eigen::Vector2d{std::cos(angle_b),  //
                                    std::sin(angle_b)};

    // TODO: this all could have been simplified.
    const Eigen::Vector2d dir = (ea + eb) * 0.5;
    auto angle = std::atan2(dir.y(), dir.x());
    if (angle < 0)
      angle += 2 * pi;

    // A zero-crossing is characterized by a negative sign between
    // consecutive intensity values.
    if (a * b < 0)
      zero_crossings.push_back(static_cast<float>(angle));
  }

  return zero_crossings;
}


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  // Input video.
  const auto video_filepath =
      argc < 2 ? "/Users/david/Desktop/Datasets/videos/sample10.mp4" : argv[1];
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();

  // Preprocessing parameters.
  const auto downscale_factor = argc < 3 ? 1 : std::stoi(argv[2]);
  const auto cornerness_adaptive_thres = argc < 4 ? 0 : std::stof(argv[3]);

  // Harris cornerness parameters.
  //
  // Blur parameter before gradient calculation.
  static const auto sigma_D = std::sqrt(sara::square(1.6f) - 1);
  // Integration domain of the second moment.
  static const auto sigma_I = 3.f;
  static const auto kappa = 0.04f;

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

    // Convert to grayscale, downsample.
    auto frame_gray = frame.convert<float>();
    if (downscale_factor > 1)
      frame_gray = sara::reduce(frame_gray, downscale_factor);

    // Calculate Harris cornerness functions.
    const auto cornerness = sara::scale_adapted_harris_cornerness(  //
        frame_gray,                                                 //
        sigma_I, sigma_D,                                           //
        kappa                                                       //
    );

    // Select the local maxima of the cornerness functions.
    static constexpr auto select = [](const sara::ImageView<float>& cornerness,
                                      const float cornerness_adaptive_thres) {
      static constexpr auto r = LukasKanadeOpticalFlowEstimator<>::patch_radius;

      const auto extrema = sara::local_maxima(cornerness);

      const auto cornerness_max = cornerness.flat_array().maxCoeff();
      const auto cornerness_thres = cornerness_adaptive_thres * cornerness_max;

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
    const auto corners = select(cornerness, cornerness_adaptive_thres);

    // Calculate the optical flow.
    flow_estimator.update_image(frame_gray);
    auto flow_vectors = std::vector<Eigen::Vector2f>{};
    if (!flow_estimator._I0.empty())
      flow_vectors = flow_estimator.estimate_flow(corners);

    // TODO: track the corners with optical flow.

    // Draw the sparse flow field.
    if (!flow_estimator._I0.empty())
    {
      for (auto i = 0u; i != corners.size(); ++i)
      {
        const auto& v = flow_vectors[i];
        if (std::isnan(v.x()) || std::isnan(v.y()))
          continue;

        const Eigen::Vector2f pa = corners[i].cast<float>() * downscale_factor;
        const Eigen::Vector2f pb = pa + flow_vectors[i] * 10;

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

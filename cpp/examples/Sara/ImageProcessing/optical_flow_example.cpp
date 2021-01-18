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

struct LukasKanadeOpticalFlowEstimator
{
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
    const auto& w = _patch_sizes.x();
    const auto& h = _patch_sizes.y();
    const auto n = w * h;

    auto A = Eigen::MatrixXf{n, 2};
    auto b = Eigen::VectorXf{n};

    const Eigen::Vector2i r = _patch_sizes / 2;
    auto i = 0;
    for (auto y = p.y() - r.y(); y <= p.y() + r.y(); ++y)
    {
      for (auto x = p.x() - r.x(); x <= p.x() + r.x(); ++x)
      {
        A(i, 0) = _grad_I(x, y).x();
        A(i, 1) = _grad_I(x, y).y();

        b(i) = -_dI_dt(x, y);

        ++i;
      }
    }

    const Eigen::Vector2f d = A.colPivHouseholderQr().solve(b);

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
  Eigen::Vector2i _patch_sizes = Eigen::Vector2i{5, 5};

  sara::Image<float> _I1;
  sara::Image<float> _I0;

  sara::Image<Eigen::Vector2f> _grad_I;
  sara::Image<float> _dI_dt;
};


int __main(int argc, char** argv)
{
  using namespace std::string_literals;

  // Input video.
  const auto video_filepath =
      argc < 2 ? "/Users/david/Desktop/Datasets/videos/sample10.mp4" : argv[1];
  sara::VideoStream video_stream(video_filepath);
  auto frame = video_stream.frame();

  // Preprocessing parameters.
  const auto downscale_factor = 3;

  // Harris cornerness parameters.
  const auto sigma_D = std::sqrt(std::pow(1.6f, 2) - 1);
  const auto sigma_I = 3.f;
  const auto kappa = 0.24f;

  // Lukas-Kanade optical flow parameters.
  const auto square_patch_radius = 3;
  auto flow_estimator = LukasKanadeOpticalFlowEstimator{};
  flow_estimator._patch_sizes =
      Eigen::Vector2i::Ones() * (2 * square_patch_radius + 1);

  sara::create_window(video_stream.sizes());
  sara::set_antialiasing();

  auto frames_read = -1;
  const auto skip = 0;
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
    auto select = [square_patch_radius](const auto& cornerness) {
      const auto extrema = sara::local_maxima(cornerness);
      const auto& r = square_patch_radius;

      auto extrema_filtered = std::vector<sara::Point2i>{};
      extrema_filtered.reserve(extrema.size());
      for (const auto& p : extrema)
      {
        const auto in_image_domain =
            (r <= p.x() && p.x() < cornerness.width() - r) &&
            (r <= p.y() && p.y() < cornerness.height() - r);

        if (cornerness(p) > 0 && in_image_domain)
          extrema_filtered.emplace_back(p);
      }
      return extrema_filtered;
    };
    const auto corners = select(cornerness);

    // Calculate the optical flow.
    flow_estimator.update_image(frame_gray);
    auto flow_vectors = std::vector<Eigen::Vector2f>{};
    if (!flow_estimator._I0.empty())
      flow_vectors = flow_estimator.estimate_flow(corners);

    // Draw the sparse flow field.
    sara::display(frame);
    if (!flow_estimator._I0.empty())
    {
      for (auto i = 0u; i != corners.size(); ++i)
      {
        const auto& v = flow_vectors[i];
        if (v.squaredNorm() < 0.25f)
          continue;

        const Eigen::Vector2f pa = corners[i].cast<float>() * downscale_factor;
        const Eigen::Vector2f pb = pa + flow_vectors[i].normalized() * 20;

        const auto& r = square_patch_radius;
        const auto& l = (2 * r + 1) * downscale_factor;
        const Eigen::Vector2i tl =
            (pa - Eigen::Vector2f::Ones() * r * downscale_factor).cast<int>();

        sara::draw_rect(tl.x(), tl.y(), l, l, sara::Red8);
        sara::draw_arrow(pa, pb, sara::Red8, 2);
      }
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

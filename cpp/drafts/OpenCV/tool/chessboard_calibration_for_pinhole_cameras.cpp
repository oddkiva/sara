#include <opencv2/opencv.hpp>

#include <DO/Sara/Core/PhysicalQuantities.hpp>
#include <DO/Sara/Core/TicToc.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry/Resectioning/HartleyZisserman.hpp>
#include <DO/Sara/VideoIO.hpp>

// #include <ceres/ceres.h>
// #include <ceres/rotation.h>

#include <opencv2/core/eigen.hpp>

#include "../HomographyDecomposition.hpp"
#include "../HomographyEstimation.hpp"


namespace sara = DO::Sara;

using sara::operator""_cm;


inline auto init_K(int w, int h) -> Eigen::Matrix3d
{
  const auto d = static_cast<double>(std::max(w, h));
  const auto f = 0.5 * d;

  // clang-format off
  const auto K = (Eigen::Matrix3d{} <<
                  f, 0, w * 0.5,
                  0, f, h * 0.5,
                  0, 0,       1).finished();
  // clang-format on

  return K;
}

GRAPHICS_MAIN()
{
// #define LUXVISION
#define GOPRO4
#define GOPRO7_WIDE
#define GOPRO7_SUPERVIEW
  // #define GOPRO4
  auto video_stream = sara::VideoStream{
#ifdef LUXVISION
      "/media/Linux "
      "Data/ha/safetytech/210330_FishEye/calibration_luxvision_cameras/"
      "checkboard_luxvision_1.MP4"
#elif defined(GOPRO4)
      "/home/david/Desktop/calibration/gopro-hero4/chessboard.mp4"
#elif defined(GOPRO7_WIDE)
      "/home/david/Desktop/calibration/gopro-hero-black-7/wide/GH010052.MP4"
#else
      "/home/david/Desktop/calibration/gopro-hero-black-7/superview/"
      "GH010053.MP4"
#endif
  };
  auto frame = video_stream.frame();

#ifdef GOPRO4
  static const auto pattern_size = Eigen::Vector2i{7, 5};
  static constexpr auto square_size = 3._cm;
#elif defined(GOPRO7_WIDE) || defined(GOPRO7_SUPERVIEW)
  static const auto pattern_size = Eigen::Vector2i{7, 9};
  static constexpr auto square_size = 2._cm;
#else
  static const auto pattern_size = Eigen::Vector2i{7, 12};
  static constexpr auto square_size = 7._cm;
#endif
  auto chessboard = sara::OpenCV::Chessboard(pattern_size, square_size.value);

  sara::create_window(frame.sizes());
  sara::set_antialiasing();

  auto K = init_K(frame.width(), frame.height());

  while (video_stream.read())
  {
    sara::tic();
    const auto corners_found = chessboard.detect(frame);
    sara::toc("Chessboard corner detection");

    if (corners_found)
    {
      draw_chessboard(frame, chessboard);

      const Eigen::Matrix3d H = estimate_H(chessboard).normalized();
      auto Rs = std::vector<Eigen::Matrix3d>{};
      auto ts = std::vector<Eigen::Vector3d>{};
      auto ns = std::vector<Eigen::Vector3d>{};

      // decompose_H_faugeras(H, K, Rs, ts, ns);
      decompose_H_RQ_factorization(H, K, Rs, ts, ns);
      const auto ret = static_cast<int>(Rs.size());

      auto frame_copy = sara::Image<sara::Rgb8>{frame};

      for (auto i = 0; i < ret; ++i)
      {
        auto Hr = Eigen::Matrix3f{};
        Hr.col(0) = Rs[i].col(0).cast<float>();
        Hr.col(1) = Rs[i].col(1).cast<float>();
        Hr.col(2) = ts[i].cast<float>();
        Hr = (K.cast<float>() * Hr).normalized();
        // Hr = (K * (Rs[i] + ts[i] *
        // ns[i].transpose())).cast<float>().normalized();

        // SARA_DEBUG << "\nRi =\n" << Rs[i] << std::endl;
        // SARA_DEBUG << "\nti =\n" << ts[i] << std::endl;
        // SARA_DEBUG << "\nni =\n" << ns[i] << std::endl;
        SARA_DEBUG << i << "\n";
        SARA_DEBUG << "H =\n" << H << "\n";
        SARA_DEBUG << "H_rec_i =\n" << Hr << "\n\n";

        const auto a = chessboard(0, 0);
        const auto b = chessboard(0, 1);
        const auto c = chessboard(1, 0);

        static const auto red = sara::Rgb8{167, 0, 0};
        sara::draw_arrow(frame_copy, a, b, red, 6);
        sara::draw_circle(frame_copy, a, 5.f, red, 6);
        sara::draw_circle(frame_copy, b, 5.f, red, 6);

        static const auto green = sara::Rgb8{89, 216, 26};
        sara::draw_arrow(frame_copy, a, c, green, 6);
        sara::draw_circle(frame_copy, a, 5.f, green, 6);
        sara::draw_circle(frame_copy, c, 5.f, green, 6);

        for (auto i = 0; i < chessboard.height(); ++i)
        {
          for (auto j = 0; j < chessboard.width(); ++j)
          {
            const Eigen::Vector2f x = chessboard(i, j);

            const Eigen::Vector3f X = chessboard.point_3d(i, j).homogeneous();
            const Eigen::Vector2f x2 = (Hr * X).hnormalized();

            sara::draw_circle(frame_copy, x, 3.f, sara::Cyan8, 3);
            sara::draw_circle(frame_copy, x2, 3.f, sara::Magenta8, 3);
          }
        }
      }

      sara::display(frame_copy);
    }
  }

  return 0;
}

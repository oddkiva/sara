#pragma once

#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>

#include <DO/Shakti/Cuda/FeatureDetectors/ScaleSpaceExtremum.hpp>

#include <drafts/Taskflow/SafeQueue.hpp>


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;
namespace sc = shakti::Cuda;
namespace scg = sc::Gaussian;


struct DisplayTask
{
  sara::Image<sara::Rgb8> image;
  sc::HostExtrema data;
  int index = -1;

  inline DisplayTask() = default;

  inline DisplayTask(sara::Image<sara::Rgb8> im,  //
                     sc::HostExtrema&& data,      //
                     int id)
    : image{std::move(im)}
    , data{std::move(data)}
    , index{id}
  {
  }

  inline DisplayTask(const DisplayTask& task) = default;

  inline DisplayTask(DisplayTask&& task) noexcept
    : image{std::move(task.image)}
    , data{std::move(task.data)}
    , index{task.index}
  {
  }

  inline ~DisplayTask() = default;

  inline auto run() -> void
  {
    if (index == -1 || image.data() == nullptr)
      return;

    const auto num_extrema = static_cast<int>(data.x.size());

#pragma omp parallel for
    for (auto k = 0; k < num_extrema; ++k)
    {
      const auto& x = data.x[k];
      const auto& y = data.y[k];
      const auto& s = data.s[k];
      const auto& type = data.types[k];
      static constexpr auto sqrt_2 = static_cast<float>(M_SQRT2);
      const auto r = s * sqrt_2;

      if (type == 1)
        sara::draw_circle(image, x, y, r, sara::Red8, 3);
      else if (type == -1)
        sara::draw_circle(image, x, y, r, sara::Blue8, 3);
    }

    sara::draw_text(image, 100, 50, sara::format("Frame: %d", index),
                    sara::White8, 30);
    sara::draw_text(image, 100, 100, sara::format("#Extrema: %d", num_extrema),
                    sara::White8, 30);


    sara::display(image);

    std::cout << "[" << index << "] " << num_extrema << " keypoints"
              << std::endl;
  }
};

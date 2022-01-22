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
    // SARA_CHECK(data.x.size());
    // SARA_CHECK(data.y.size());
    // SARA_CHECK(data.types.size());

    sara::draw_text(image, 100, 50, sara::format("Frame: %d", index),
                    sara::White8, 30);
    sara::draw_text(image, 100, 100, sara::format("#Extrema: %d", num_extrema),
                    sara::White8, 30);

#pragma omp parallel for
    for (auto k = 0; k < num_extrema; ++k)
    {
      const auto& x = data.x[k];
      const auto& y = data.y[k];
      const auto& s = data.s[k];
      const auto& value = data.values[k];
      const auto& refined = data.refined[k];
      const auto& type = data.types[k];
      SARA_CHECK(x);
      SARA_CHECK(y);
      SARA_CHECK(s);
      SARA_CHECK(value);
      SARA_CHECK(int(type));
      SARA_CHECK(int(refined));

      if (type == 1)
        sara::draw_circle(image, x, y, 4, sara::Red8, 3);
      else if (type == -1)
        sara::draw_circle(image, x, y, 4, sara::Blue8, 3);
    }

    sara::display(image);

    std::cout << "[" << index << "] " << num_extrema << " keypoints"
              << std::endl;
  }
};

#pragma once

#include <DO/Sara/Graphics.hpp>

namespace DO::Sara {

  template <typename T = float>
  struct DisplayTask
  {
    Image<T> image;
    int index = -1;

    inline DisplayTask() = default;

    inline DisplayTask(Image<T> im, int id)
      : image{std::move(im)}
      , index{id}
    {
    }

    inline DisplayTask(const DisplayTask& task) = default;

    inline DisplayTask(DisplayTask&& task)
      : image{std::move(task.image)}
      , index{task.index}
    {
    }

    inline ~DisplayTask() = default;

    inline auto run() -> void
    {
      if (index == -1 || image.data() == nullptr)
        return;
      auto image_rgb = image.template convert<Rgb8>();
      draw_text(image_rgb, 100, 50, std::to_string(index), White8,
                      30);
      display(image_rgb);
      std::cout << "Showing frame " << index << std::endl;
    }
  };

}  // namespace DO::Sara

#pragma once

#include <DO/Sara/Core/Timer.hpp>


struct Time
{
  void update()
  {
    last_frame = current_frame;
    current_frame = static_cast<float>(timer.elapsed_ms());
    delta_time = current_frame - last_frame;
  }

  DO::Sara::Timer timer;
  float delta_time = 0.f;
  float last_frame = 0.f;
  float current_frame = 0.f;
};

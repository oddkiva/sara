#pragma once

#include "AsyncDisplayTask.hpp"


inline auto view_octave(sc::Octave<float>& octave, bool& quit,
                        bool color_rescale = false) -> void
{
  quit = false;

  auto h_oct = sara::Image<float, 3>{octave.width(), octave.height(),
                                     octave.scale_count()};
  octave.array().copy_to(h_oct);

  const auto h_oct_tensor = sara::tensor_view(h_oct);
  for (auto s = 0; s < octave.scale_count(); ++s)
  {
    const auto slice_s = sara::image_view(h_oct_tensor[s]);
    if (color_rescale)
      sara::display(sara::color_rescale(slice_s));
    else
      sara::display(slice_s);
    quit = sara::get_key() == sara::KEY_ESCAPE;
  }
}

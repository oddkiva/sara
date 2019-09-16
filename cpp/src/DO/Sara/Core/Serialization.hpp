#pragma once

#include <DO/Sara/Core/MultiArray/MultiArray.hpp>
#include <DO/Sara/Core/MultiArray/MultiArrayView.hpp>

#include <iostream>


namespace boost { namespace serialization {

  template <class Archive, typename T, int N, int O>
  void serialize(Archive& ar,                       //
                 DO::Sara::MultiArray<T, N, O>& t,  //
                 const unsigned int /* version */)
  {
    auto sizes = t.sizes();
    for (auto i = 0u; i != sizes.size(); ++i)
      ar & sizes[i];

    if (t.sizes() != sizes)
      t.resize(sizes);

    for (auto i = 0u; i != t.size(); ++i)
      ar & t.data()[i];
  }

} /* namespace serialization */
} /* namespace boost */

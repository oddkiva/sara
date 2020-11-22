#pragma once

#include <pybind11/numpy.h>


template <typename Sequence>
inline auto as_pyarray(Sequence&& seq)
    -> pybind11::array_t<typename Sequence::value_type>
{
  auto seq_ptr = new Sequence{std::move(seq)};
  auto capsule = pybind11::capsule{
      seq_ptr,                                                //
      [](void* p) { delete reinterpret_cast<Sequence*>(p); }  //
  };

  return pybind11::array{
      seq_ptr->size(),  // shape of array
      seq_ptr->data(),  // c-style contiguous strides for Sequence
      capsule           // numpy array references this parent
  };
}

template <typename Sequence>
inline auto to_pyarray(const Sequence& seq)
    -> pybind11::array_t<typename Sequence::value_type>
{
  return pybind11::array{seq.size(), seq.data()};
}

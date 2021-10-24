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

#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Features/Feature.hpp>
#include <DO/Sara/Features/KeypointList.hpp>


PYBIND11_MAKE_OPAQUE(DO::Sara::KeypointList<DO::Sara::OERegion, float>);
PYBIND11_MAKE_OPAQUE(std::vector<DO::Sara::OERegion>);


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


template <typename T, int N>
auto wrap_tensor_class(pybind11::module& m, const std::string& name)
{
  namespace py = pybind11;
  namespace sara = DO::Sara;

  auto to_vector = [](const auto& vec) {
    auto v = std::vector<py::ssize_t>(vec.size());
    std::transform(vec.data(), vec.data() + vec.size(),
                   [](const auto& x) { return static_cast<py::ssize_t>(x); });
  };

  py::class_<sara::Tensor_<T, N>>(m, name, py::buffer_protocol())
      .def_buffer([&](sara::Tensor_<T, N>& m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                           /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            py::format_descriptor<T>::format(), /* Python struct-style format
                                                   descriptor */
            N,                                  /* Number of dimensions */
            to_vector(m.sizes()),               /* Buffer dimensions */
            to_vector((m.strides() * sizeof(T))
                          .eval()) /* Strides (in bytes) for each index */
        );
      });
}

template <typename T>
inline auto to_image_view(pybind11::array_t<T> image)
{
  namespace sara = DO::Sara;

  if (image.ndim() != 2)
    throw std::runtime_error{"Invalid image shape!"};

  const auto height = static_cast<int>(image.shape(0));
  const auto width = static_cast<int>(image.shape(1));
  auto data = const_cast<T*>(image.data());
  auto imview = sara::ImageView<T, 2>{data, {width, height}};
  return imview;
}

template <typename T>
inline auto to_interleaved_rgb_image_view(pybind11::array_t<T> image)
{
  namespace sara = DO::Sara;

  if (image.ndim() != 3 || image.shape(2) != 3)  //
    throw std::runtime_error{"Invalid image shape!"};

  using Pixel = sara::Pixel<T, sara::Rgb>;
  const auto height = static_cast<int>(image.shape(0));
  const auto width = static_cast<int>(image.shape(1));
  auto data = const_cast<T*>(image.data());
  auto imview = sara::ImageView<Pixel, 2>{reinterpret_cast<Pixel*>(data),
                                          {width, height}};
  return imview;
}


namespace pybind11::detail {

  template <typename T>
  struct type_caster<DO::Sara::TensorView_<T, 2>>
  {
  public:
    using TensorView2d = DO::Sara::TensorView_<T, 2>;
    PYBIND11_TYPE_CASTER(TensorView2d, _("DO::Sara::TensorView_<T, 2>"));

    // Cast a NumPy array to C++ DO::Sara::TensorView_ object.
    bool load(pybind11::handle src, bool convert)
    {
      if (!convert and !pybind11::array_t<T>::check_(src))
        return false;

      // Try converting a generic Python object to a NumPy array object.
      auto buffer =
          pybind11::array_t<T, pybind11::array::c_style |
                                   pybind11::array::forcecast>::ensure(src);
      if (!buffer)
        return false;

      value.swap(DO::Sara::TensorView_<T, 2>{
          const_cast<T*>(buffer.data()),
          Eigen::Vector2i(buffer.shape()[0], buffer.shape()[1])});

      return true;
    }

    // Cast a C++ DO::Sara::TensorView_<T, 2> object to a NumPy array.
    static pybind11::handle cast(const DO::Sara::TensorView_<T, 2>& src,
                                 pybind11::return_value_policy,
                                 pybind11::handle)
    {
      std::vector<size_t> shape(2);
      std::vector<size_t> strides(2);

      shape[0] = src.size(0);
      shape[1] = src.size(1);

      strides[0] = src.stride(0) * sizeof(T);
      strides[1] = src.stride(1) * sizeof(T);

      pybind11::array a{std::move(shape), std::move(strides), src.data()};

      return a.release();
    }
  };

  template <typename T>
  struct type_caster<DO::Sara::ImageView<T>>
  {
  public:
    PYBIND11_TYPE_CASTER(DO::Sara::ImageView<T>, _("DO::Sara::ImageView<T>"));

    // Cast a NumPy array to C++ DO::Sara::Image object.
    bool load(pybind11::handle src, bool convert)
    {
      if (!convert and !pybind11::array_t<T>::check_(src))
        return false;

      // Try converting a generic Python object to a NumPy array object.
      auto buffer =
          pybind11::array_t<T, pybind11::array::c_style |
                                   pybind11::array::forcecast>::ensure(src);
      if (!buffer)
        return false;

      if (buffer.ndim() != 2)
        return false;

      value.swap(DO::Sara::ImageView<T>{
          const_cast<T*>(buffer.data()),
          Eigen::Vector2i(buffer.shape()[1], buffer.shape()[0])});

      return true;
    }

    // Cast a C++ DO::Sara::Image object to a NumPy array.
    static pybind11::handle cast(const DO::Sara::ImageView<T>& src,
                                 pybind11::return_value_policy,
                                 pybind11::handle)
    {
      std::vector<size_t> shape(2);
      std::vector<size_t> strides(2);

      shape[0] = src.height();
      shape[1] = src.width();

      strides[0] = sizeof(T);
      strides[1] = shape[1] * sizeof(T);

      pybind11::array a{std::move(shape), std::move(strides), src.data()};

      return a.release();
    }
  };

}  // namespace pybind11::detail

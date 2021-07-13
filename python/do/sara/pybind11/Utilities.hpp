#pragma once

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/Core/Tensor.hpp>


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

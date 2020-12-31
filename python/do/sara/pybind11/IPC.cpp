// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include "IPC.hpp"

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <iostream>


namespace bip = boost::interprocess;
namespace py = pybind11;


template <typename T>
using ipc_allocator =
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template <typename T>
using ipc_vector = bip::vector<T, ipc_allocator<T>>;


template <class T>
auto to_py_list(const ipc_vector<T>& v) -> std::vector<T>
{
  auto l = std::vector<T>{};
  for (const auto& c : v)
    l.push_back(c);
  return l;
}


// class IpcMedium
// {
// public:
//   IpcMedium(const std::string& segment_name)
//     : _segment{bip::open_only, segment_name.c_str()}
//   {
//   }
//
//   py::array_t<float> tensor(const std::string& name)
//   {
//     const auto image_shape_name = name + "_shape";
//     const auto image_data_name = name + "_data";
//
//     auto image_shape =
//         _segment.find<ipc_vector<int>>(image_shape_name.c_str()).first;
//     auto image_data =
//         _segment.find<ipc_vector<float>>(image_data_name.c_str()).first;
//
//     const auto shape = to_py_list(*image_shape);
//     const auto strides =
//         bp::make_tuple(sizeof(float) * (*image_shape)[1], sizeof(float));
//
//     return np::from_data(image_data->data(), np::dtype::get_builtin<float>(),
//                          shape, strides, bp::object());
//   }
//
// private:
//   bip::managed_shared_memory _segment;
// };
//
//
// void expose_ipc()
// {
//   bp::class_<IpcMedium, boost::noncopyable>("IpcMedium",
//                                             bp::init<const std::string&>())
//       .def("tensor", &IpcMedium::tensor);
// }

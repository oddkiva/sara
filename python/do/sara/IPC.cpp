#include "IPC.hpp"
#include "Numpy.hpp"

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>


namespace bip = boost::interprocess;
namespace bp = boost::python;


template <typename T>
using ipc_allocator =
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template <typename T>
using ipc_vector = bip::vector<T, ipc_allocator<T>>;


template <class T>
bp::list to_py_list(const ipc_vector<T>& v)
{
  auto l = bp::list{};
  for (const auto& c : v)
    l.append(c);
  return l;
}


namespace np = bp::numpy;


class IpcMedium
{
public:
  IpcMedium(const std::string& segment_name)
    : _segment{bip::open_only, segment_name.c_str()}
  {
  }

  bip::managed_shared_memory _segment;

  int  _image_batch_filling_iter = -1;
  int  _image_batch_processing_iter = -1;
  int  _num_iter = -1;
  bool _terminate_processing = false;

  //ipc_vector<int> *image_shape;
  //ipc_vector<float> *image_data;

  //auto image_view() const -> DO::Sara::MultiArrayView<float, 2>
  //{
  //  return DO::Sara::MultiArrayView<float, 2>{
  //      image_data->data(), {(*image_shape)[0], (*image_shape)[1]}};
  //}

  bp::list image_shape(const std::string& name)
  {
    auto image_shape = _segment.find<ipc_vector<int>>(name.c_str()).first;
    return to_py_list(*image_shape);
  }

  //float * image_data(const std::string& name)
  //{
  //  auto image_data = _segment.find<ipc_vector<float>>(name.c_str()).first;
  //  return image_data->data();
  //}

  np::ndarray image_data(const std::string& name)
  {
    auto image_data = _segment.find<ipc_vector<float>>(name.c_str()).first;
    return np::from_data(image_data->data(), np::dtype::get_builtin<float>(),
                         bp::make_tuple(image_data->size()),
                         bp::make_tuple(sizeof(float)),
                         bp::object());
  }
};


void expose_ipc()
{
  bp::class_<IpcMedium, boost::noncopyable>("IpcMedium",
                                            bp::init<const std::string&>())
      .def("image_shape", &IpcMedium::image_shape)
      .def("image_data", &IpcMedium::image_data);
}

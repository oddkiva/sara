#include "IPC.hpp"

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <boost/python.hpp>


namespace bip = boost::interprocess;
namespace bp = boost::python;


template <typename T>
using ipc_allocator =
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template <typename T>
using ipc_vector = bip::vector<T, ipc_allocator<T>>;


class IpcMedium
{
public:
  IpcMedium() = default;

  IpcMedium(const std::string& segment_name)
    : _segment{bip::open_only, segment_name.c_str()}
  {
  }

  int image_batch_fill_iter()
  {
    return _image_batch_filling_iter;
  }

  int image_batch_processing_iter()
  {
    return _image_batch_processing_iter;
  }

  void set_image_batch_processing_iter(int iter)
  {
    _image_batch_processing_iter = iter;
  }

  bip::managed_shared_memory _segment;

  bip::interprocess_mutex _mutex;
  bip::interprocess_condition _cond_image_batch_refilled;
  bip::interprocess_condition _cond_image_batch_processed;
  bip::interprocess_condition _cond_terminate_processing;

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
};


class Lock
{
public:
  Lock(IpcMedium& medium)
    : _lock(medium._mutex)
  {
  }

  void lock() { _lock.lock(); }
  void unlock() { _lock.unlock(); }
  void release() { _lock.release(); }

  bip::scoped_lock<bip::interprocess_mutex> _lock;
};


void wait_for_image_batch_refill(IpcMedium& medium, Lock& lock)
{
  medium._cond_image_batch_refilled.wait(lock._lock);
}

void wait_for_termination_signal(IpcMedium& medium, Lock& lock)
{
  medium._cond_terminate_processing.wait(lock._lock);
}

void notify_image_batch_processed(IpcMedium& medium)
{
  medium._cond_image_batch_processed.notify_one();
}


void expose_ipc()
{
  bp::class_<Lock, boost::noncopyable>("Lock", bp::init<IpcMedium&>())
      .def("lock", &Lock::lock)
      .def("unlock", &Lock::unlock)
      .def("release", &Lock::release);

  bp::class_<IpcMedium, boost::noncopyable>("IpcMedium",
                                            bp::init<const std::string&>())
      .def("image_batch_fill_iter", &IpcMedium::image_batch_fill_iter)
      .def("image_batch_processing_iter",
           &IpcMedium::image_batch_processing_iter)
      .def("set_image_batch_processing_iter",
           &IpcMedium::set_image_batch_processing_iter);

  bp::def("wait_for_image_batch_refill", &wait_for_image_batch_refill);
  bp::def("wait_for_termination_signal", &wait_for_termination_signal);
  bp::def("notify_image_batch_processed", &notify_image_batch_processed);
}

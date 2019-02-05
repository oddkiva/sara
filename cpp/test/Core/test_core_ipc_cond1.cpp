#include <DO/Sara/Core/MultiArray.hpp>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <iostream>
#include <thread>

#include <zmq.h>


namespace sara = DO::Sara;
namespace bip = boost::interprocess;


template <typename T>
using ipc_allocator =
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template <typename T>
using ipc_vector = bip::vector<T, ipc_allocator<T>>;


struct Message
{
  bip::interprocess_mutex mutex;
  bip::interprocess_condition cond_image_batch_refilled;
  bip::interprocess_condition cond_image_batch_processed;
  bip::interprocess_condition cond_terminate_processing;

  int image_batch_filling_iter = -1;
  int image_batch_processing_iter = -1;
  int num_iter = -1;
  bool terminate_processing = false;
};


template <typename T, int N>
auto construct_tensor(bip::managed_shared_memory& segment,
                      const std::string& name,
                      const sara::Matrix<int, N, 1>& shape)
    -> sara::MultiArrayView<T, N, sara::RowMajor>
{
  using dtype = T;

  bip::allocator<int, bip::managed_shared_memory::segment_manager>
      int_allocator{segment.get_segment_manager()};

  bip::allocator<dtype, bip::managed_shared_memory::segment_manager>
      dtype_allocator{segment.get_segment_manager()};

  const auto tensor_shape_name = name + "_shape";
  const auto tensor_data_name = name + "_data";

  // Allocate image data in the memory segment.
  segment.construct<ipc_vector<int>>(tensor_shape_name.c_str())(
      shape.data(), shape.data() + shape.size(), int_allocator);
  const auto tensor_size = std::accumulate(
      shape.data(), shape.data() + shape.size(), 1, std::multiplies<int>());
  auto tensor_vec = segment.construct<ipc_vector<T>>(tensor_data_name.c_str())(
      tensor_size, static_cast<T>(1), dtype_allocator);
  std::cout << "tensor_vec.size() = " << tensor_vec->size() << std::endl;

  return sara::MultiArrayView<T, N, sara::RowMajor>{tensor_vec->data(), shape};
}

template <typename T, int N>
auto find_tensor(bip::managed_shared_memory& segment,
                 const std::string& name)
  -> sara::MultiArrayView<T, N, sara::RowMajor>
{
  std::cout << "Find tensor name = " << name << std::endl;
  const auto tensor_shape_name = name + "_shape";
  const auto tensor_data_name = name + "_data";
  auto tensor_shape =
      segment.find<ipc_vector<int>>(tensor_shape_name.c_str()).first;
  auto tensor_data =
      segment.find<ipc_vector<T>>(tensor_data_name.c_str()).first;

  auto tensor_shape_vector = sara::Matrix<int, N, 1>{};
  std::copy(tensor_shape->begin(), tensor_shape->end(),
            tensor_shape_vector.data());
  std::cout << tensor_shape_vector.matrix().transpose() << std::endl;

  return sara::MultiArrayView<float, 2, sara::RowMajor>{tensor_data->data(),
                                                        tensor_shape_vector};
}

template <typename T>
auto destroy_tensor(bip::managed_shared_memory& segment,
                    const std::string& name)  //
    -> void
{
  const auto tensor_shape_name = name + "_shape";
  const auto tensor_data_name = name + "_data";
  segment.destroy<ipc_vector<int>>(tensor_shape_name.c_str());
  segment.destroy<ipc_vector<T>>(tensor_data_name.c_str());
}


int main(int argc, char** argv)
{
  if (argc == 1)
  {
    struct SharedMemoryRemover
    {
      SharedMemoryRemover()
      {
        bip::shared_memory_object::remove("MySharedMemory");
      }
      ~SharedMemoryRemover()
      {
        bip::shared_memory_object::remove("MySharedMemory");
      }
    } remover;

    bip::managed_shared_memory segment{bip::create_only, "MySharedMemory",
                                       64 * 1024 * 1024 * 3 * sizeof(float)};

    bip::allocator<Message, bip::managed_shared_memory::segment_manager>
      message_allocator{segment.get_segment_manager()};

    bip::allocator<float, bip::managed_shared_memory::segment_manager>
        float_allocator{segment.get_segment_manager()};

    const auto num_iter = 10;

    // Synchronisation between processes.
    auto message = segment.construct<Message>("message")();
    message->num_iter = num_iter;

    // Allocate image in the memory segment.
    auto image =
        construct_tensor<float, 2>(segment, "image", DO::Sara::Vector2i{3, 4});

    // Allocate image descriptors data in the memory segment.
    auto image_descriptors = segment.construct<ipc_vector<float>>(
        "image_descriptors")(128, 0.f, float_allocator);

    // Start Process 2 by running a command in an asynchronous way.
    //
    // N.B.: Mutex and scoped_lock does not seem to work in Python when we
    // expose their API in Python...
    //auto command = std::string{"python "
    //                           "/home/david/GitHub/DO-CV/sara-build-Debug/lib/"
    //                           "do/sara/test/test_ipc_example.py"};
    //
    // SOLUTION: I chose to use zeromq to have synchronized communication
    // between C++ and Python.
    const auto command = std::string{argv[0]} + " 1";
    std::thread t([&command]() {
      std::cout << "running " << command << std::endl;
      std::system(command.c_str());
    });
    t.detach();

    // Loop where the two processes communicate.
    for (int i = 0; i < num_iter; ++i)
    {
      std::cout << "[Process 1] Iteration " << i << "\n";  // std::endl;
      bip::scoped_lock<bip::interprocess_mutex> lock(message->mutex);

      // Fill with new image data.
      std::cout << "[Process 1] Refilling image data" << std::endl;
      image.flat_array().fill(i);

      std::cout << "[Process 1] Refilled image data" << std::endl;
      message->image_batch_filling_iter = i;
      message->cond_image_batch_refilled.notify_one();

      // Wait until the image is processed.
      if (message->image_batch_processing_iter != i)
      {
        // std::cout << "[Process 1] Waiting for Process 2 to complete"
        //  << std::endl;
        message->cond_image_batch_processed.wait(lock);
      }

      // Print the calculated descriptors.
      std::cout << "[Process 1] Process 2 calculated descriptors" << std::endl;
      for (auto i = 0; i < 10; ++i)
        std::cout << (*image_descriptors)[i] << " ";
      std::cout << std::endl << std::endl;

      if (message->image_batch_processing_iter == num_iter - 1)
      {
        std::cout << "[Process 1] Notifying Process 2 to terminate"
                  << std::endl;
        message->cond_terminate_processing.notify_one();
      }
    };

    destroy_tensor<float>(segment, "image");
    segment.destroy<ipc_vector<float>>("image_descriptors");
    segment.destroy<Message>("message");
  }
  else
  {
    std::cout << "Running child process" << std::endl;

    bip::managed_shared_memory segment{bip::open_only, "MySharedMemory"};

    auto image_view = find_tensor<float, 2>(segment, "image");

    auto image_descriptors =
        segment.find<ipc_vector<float>>("image_descriptors").first;
    auto message = segment.find<Message>("message").first;


    while (true)
    {
      bip::scoped_lock<bip::interprocess_mutex> lock(message->mutex);
      if (message->image_batch_filling_iter <=
          message->image_batch_processing_iter)
      {
        std::cout << "[Process 2] Waiting for Process 1 to refill image data"
                  << std::endl;
        message->cond_image_batch_refilled.wait(lock);
      }

      const auto value = image_view.flat_array()[0];
      std::cout << "[Process 2] image_view =\n" << image_view.matrix() << std::endl;

      std::cout << "[Process 2] Calculating descriptors" << std::endl;
      std::fill(image_descriptors->begin(), image_descriptors->end(), value);

      std::cout << "[Process 2] Notifying Process 1 that image processing is "
                   "finished"
                << std::endl;
      message->image_batch_processing_iter = message->image_batch_filling_iter;
      message->cond_image_batch_processed.notify_one();


      if (message->image_batch_processing_iter == message->num_iter - 1)
      {
        std::cout << "[Process 2] Waiting for Process 1 to signal termination"
                  << std::endl;
        message->cond_terminate_processing.wait(lock);

        std::cout << "[Process 2] Received signal from Process 1 to "
                     "terminate process"
                  << std::endl;
        break;
      }
    }
  }

  return 0;
}

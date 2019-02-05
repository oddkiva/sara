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


namespace bip = boost::interprocess;


template <typename T>
using ipc_allocator =
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template <typename T>
using ipc_vector = bip::vector<T, ipc_allocator<T>>;


struct Message
{
  int image_batch_filling_iter = -1;
  int image_batch_processing_iter = -1;
  int num_iter = -1;
  bool terminate_processing = false;
};


int main(int argc, char** argv)
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

  bip::allocator<int, bip::managed_shared_memory::segment_manager>
      int_allocator{segment.get_segment_manager()};

  bip::allocator<float, bip::managed_shared_memory::segment_manager>
      float_allocator{segment.get_segment_manager()};

  const auto num_iter = 10;

  // Synchronisation between processes.
  auto message = segment.construct<Message>("message")();
  message->num_iter = num_iter;

  // Allocate image data in the memory segment.
  segment.construct<ipc_vector<int>>("image_shape")(
      std::initializer_list<int>{3, 4}, int_allocator);
  auto image_data = segment.construct<ipc_vector<float>>("image_data")(
      3 * 4, 0.f, float_allocator);

  // Allocate image descriptors data in the memory segment.
  auto image_descriptors = segment.construct<ipc_vector<float>>(
      "image_descriptors")(128, 0.f, float_allocator);


  // Start Process 2 by running a command in an asynchronous way.
  auto command = std::string{"python "
                             "/home/david/GitHub/DO-CV/sara-build-Debug/lib/"
                             "do/sara/test/test_ipc_example.py"};
  std::thread t([&command]() {
    std::cout << "running " << command << std::endl;
    std::system(command.c_str());
  });
  t.detach();

  auto context = zmq_ctx_new();
  auto responder = zmq_socket(context, ZMQ_REP);
  auto rc = zmq_bind(responder, "tcp://*:5555");
  if (rc != 0)
  {
    std::cout << "rc value is not 0!" << std::endl;
    return 1;
  }


  constexpr std::uint8_t REFILLED_IMAGE = 0;
  constexpr std::uint8_t PROCESSED_IMAGE = 0;
  constexpr std::uint8_t TERMINATE = 0;

  for (int i = 0; i < 10; ++i)
  {
    std::cout << "[C++] Iteration " << i << "\n";  // std::endl;

    // Fill with new image data.
    std::cout << "[C++] Refilling image data" << std::endl;
    std::fill(image_data->begin(), image_data->end(), i);

    // Notify that we refilled the image batch.
    std::cout << "[C++] Notify Python" << std::endl;
    zmq_send(responder, &REFILLED_IMAGE, 1, 0);

    std::uint8_t buffer;
    zmq_recv(responder, &buffer, 1, 0);
    std::cout << "[C++] Received " << buffer << std::endl;
    
    std::cout << "[C++] ";
    for (int i = 0; i < 10; ++i)
      std::cout << (*image_data)[i] << " ";
    std::cout << std::endl;
  }


  segment.destroy<ipc_vector<int>>("image_shape");
  segment.destroy<ipc_vector<float>>("image_data");
  segment.destroy<ipc_vector<float>>("image_descriptors");
  segment.destroy<Message>("message");

  return 0;
}

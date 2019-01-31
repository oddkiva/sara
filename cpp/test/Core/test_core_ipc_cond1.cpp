#include <DO/Sara/Core/MultiArray.hpp>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <iostream>
#include <thread>


namespace bip = boost::interprocess;


template <typename T>
using ipc_allocator =
    bip::allocator<T, bip::managed_shared_memory::segment_manager>;

template <typename T>
using ipc_vector = bip::vector<T, ipc_allocator<T>>;


struct Message {
  bip::interprocess_mutex mutex;
  bip::interprocess_condition cond_image_batch_refilled;
  bip::interprocess_condition cond_image_batch_processed;
  bip::interprocess_condition cond_terminate_processing;

  int image_batch_filling_iter = -1;
  int image_batch_processing_iter = -1;
  int num_iter = -1;

  //ipc_vector<int> *image_shape;
  //ipc_vector<float> *image_data;

  //auto image_view() const -> DO::Sara::MultiArrayView<float, 2>
  //{
  //  return DO::Sara::MultiArrayView<float, 2>{
  //      image_data->data(), {(*image_shape)[0], (*image_shape)[1]}};
  //}
};



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

    bip::allocator<int, bip::managed_shared_memory::segment_manager>
        int_allocator{segment.get_segment_manager()};

    bip::allocator<float, bip::managed_shared_memory::segment_manager>
        float_allocator{segment.get_segment_manager()};

    const auto num_iter = 100000;

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
    auto command = std::string{argv[0]} + " 1";
    std::thread t([&command]() {
      std::cout << "running " << command << std::endl;
      std::system(command.c_str());
    });
    t.detach();


    // Loop where the two processes communicate.
    for (int i = 0; i < num_iter; ++i)
    {
      std::cout << "[Process 1] Iteration " << i << std::endl;
      bip::scoped_lock<bip::interprocess_mutex> lock(message->mutex);


      // Fill with new image data.
      std::cout << "[Process 1] Refilling image data" << std::endl;
      std::fill(image_data->begin(), image_data->end(), i);

      std::cout << "[Process 1] Refilled image data" << std::endl;
      message->image_batch_filling_iter = i;
      message->cond_image_batch_refilled.notify_one();


      // Wait until the image is processed.
      if (message->image_batch_processing_iter != i)
      {
        std::cout << "[Process 1] Waiting for Process 2 to complete"
          << std::endl;
        message->cond_image_batch_processed.wait(lock);
      }

      // Print the calculated descriptors.
      std::cout << "[Process 1] Process 2 calculated descriptors"
        << std::endl;
      for (auto i = 0; i < 10; ++i)
        std::cout << (*image_descriptors)[i] << " ";
      std::cout << std::endl << std::endl;

      if (message->image_batch_processing_iter == num_iter - 1)
      {
        std::cout << "[Process 1] Notifying Process 2 to terminate" << std::endl;
        message->cond_terminate_processing.notify_one();
      }
    };


    segment.destroy<ipc_vector<int>>("image_shape");
    segment.destroy<ipc_vector<float>>("image_data");
    segment.destroy<ipc_vector<float>>("image_descriptors");
    segment.destroy<Message>("message");
  }
  else
  {
    std::cout << "Running child process" << std::endl;

    bip::managed_shared_memory segment{bip::open_only, "MySharedMemory"};

    auto image_shape = segment.find<ipc_vector<int>>("image_shape").first;
    auto image_data = segment.find<ipc_vector<float>>("image_data").first;
    auto image_descriptors =
        segment.find<ipc_vector<float>>("image_descriptors").first;
    auto message = segment.find<Message>("message").first;

    auto image_view = DO::Sara::MultiArrayView<float, 2>{
        image_data->data(), {(*image_shape)[0], (*image_shape)[1]}};

    while (true)
    {
      {
        bip::scoped_lock<bip::interprocess_mutex> lock(message->mutex);
        if (message->image_batch_filling_iter <=
            message->image_batch_processing_iter)
        {
          std::cout << "[Process 2] Waiting for Process 1 to refill image data"
                    << std::endl;
          message->cond_image_batch_refilled.wait(lock);
        }

        const auto value = (*image_data)[0];

        std::cout << "[Process 2] Calculating descriptors" << std::endl;
        std::fill(image_descriptors->begin(), image_descriptors->end(), value);

        std::cout << "[Process 2] Notifying Process 1 that image processing is "
                     "finished"
                  << std::endl;
        message->image_batch_processing_iter =
            message->image_batch_filling_iter;
        message->cond_image_batch_processed.notify_one();
      }

      {
        bip::scoped_lock<bip::interprocess_mutex> lock(message->mutex);
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

  }

  return 0;
}

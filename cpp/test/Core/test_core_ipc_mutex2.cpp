#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include <iostream>


namespace bip = boost::interprocess;


struct shared_memory_log
{
  enum { NumItems = 100 };
  enum { LineSize = 100 };

  shared_memory_log() = default;

  // Mutex to protect access to the queue
  boost::interprocess::interprocess_mutex mutex;

  // Items to fill
  char items[NumItems][LineSize];
  int current_line = 0;
  bool end_a = false;
  bool end_b = false;
};


int main(int argc, char** argv)
{
  struct shm_remover
  {
    ~shm_remover() { bip::shared_memory_object::remove("MySharedMemory"); }
  } remover;

  bip::shared_memory_object shm{bip::open_only, "MySharedMemory",
                                bip::read_write};

  // Map the whole shared memory in the process.
  bip::mapped_region region{shm, bip::read_write};
  auto addr = region.get_address();

  // Construct the shared structure in memory.
  auto data = static_cast<shared_memory_log *>(addr);

  // Write some logs.
  for (int i = 0; i < 100; ++i)
  {
    bip::scoped_lock<bip::interprocess_mutex> lock(data->mutex);

    std::sprintf(
        data->items[(data->current_line++) % shared_memory_log::NumItems],
        "%s_%d", "process_b", i);

    if (i == shared_memory_log::NumItems - 1)
      data->end_b = true;
  }

  while (true)
  {
    bip::scoped_lock<bip::interprocess_mutex> lock(data->mutex);
    if (data->end_a)
      break;
  }

  return 0;
}

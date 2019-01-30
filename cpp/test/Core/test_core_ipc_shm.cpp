#include <boost/interprocess/managed_shared_memory.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>


int main(int argc, char **argv)
{
  namespace bip = boost::interprocess;

  if (argc == 1)
  {
    std::cout << "Running parent process" << std::endl;

    // Remove shared memory on construction and destruction.
    struct shm_remove
    {
      shm_remove()
      {
        bip::shared_memory_object::remove("MySharedMemory");
      }

      ~shm_remove()
      {
        bip::shared_memory_object::remove("MySharedMemory");
      }
    } remover;

    // Create a managed shared memory segment.
    bip::managed_shared_memory segment{bip::create_only, "MySharedMemory",
                                       65536 /* bytes */};
    std::cout << "- Total free memory: " << segment.get_free_memory() << std::endl;

    // Allocate a portion of the segment.
    auto free_memory = segment.get_free_memory();
    auto shptr = segment.allocate(1024);
    static_assert(std::is_same<decltype(shptr), void*>::value, "");
    std::cout << "- Free memory after allocating 1024 bytes: "
              << segment.get_free_memory() << std::endl;

    // Check invariant.
    if (free_memory <= segment.get_free_memory())
      return 1;

    // A handle from the base address can identify any byte of the shared memory
    // segment even if it is mapped in different base addresses.
    auto handle = segment.get_handle_from_address(shptr);

    std::cout << "- shptr = " << shptr << std::endl;
    std::cout << "- memory handle for shptr = " << handle << std::endl;

    std::stringstream s;
    s << argv[0] << " " << handle;
    s << std::ends;

    std::cout << "Calling the following system command: \"" << s.str() << "\"" << std::endl;

    for (int i = 0; i < 10; ++i)
    {
      // Launch child process.
      if (0 != std::system(s.str().c_str()))
        return 1;

    }

    std::cout << "Before deallocating: " << segment.get_free_memory() << std::endl;

    segment.deallocate(shptr);

    // Check memory has been freed.
    if (free_memory != segment.get_free_memory())
      return 1;

    std::cout << "After deallocating: " << segment.get_free_memory() << std::endl;
    std::cout << "Deallocated memory allocated at shptr" << std::endl;
  }
  else
  {
    std::cout << "Running child process from command: \"";
    for (int a = 0; a < argc; ++a)
      std::cout << argv[a] << ((a == argc - 1) ? "" : " ");
    std::cout << "\"" << std::endl;

    bip::managed_shared_memory segment{bip::open_only, "MySharedMemory"};

    auto handle = bip::managed_shared_memory::handle_t{0};

    std::stringstream s; s << argv[1]; s >> handle;

    auto shptr = segment.get_address_from_handle(handle);
    std::cout << "shptr address from handle = " << shptr << std::endl;
  }

  return 0;
}

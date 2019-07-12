#include <hpx/hpx_init.hpp>
#include <hpx/parallel/algorithm.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/include/iostreams.hpp>


int hpx_main(int, char **)
{
  hpx::cout << "Hello world!\n" << hpx::flush;

  auto data = std::vector<int>(int(std::sqrt(std::numeric_limits<int>::max())));
  std::iota(std::begin(data), std::end(data), 0);
  hpx::parallel::for_each(hpx::parallel::execution::par, std::begin(data),
                          std::end(data), [](auto& val) { val *= val; });

  for (const auto& x: data)
    hpx::cout << x << hpx::endl;

  return hpx::finalize();
}

int main(int argc, char** argv)
{
  return hpx::init(argc, argv);
}

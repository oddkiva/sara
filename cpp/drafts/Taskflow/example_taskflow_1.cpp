#include <thread>
#include <vector>

#include <boost/circular_buffer.hpp>


auto m = std::mutex{};
auto cv = std::condition_variable{};
auto wait_for_something = false;
auto quit = false;

auto queue = std::queue<std::vector<float>>{};

auto consumer()
{
  while (true)
  {
    std::unique_lock<std::mutex> auto lock{m};

    if (queue.empty())
      continue;

    auto& image = 

    queue.pop()

    if (quit)
      break;
  }
}

auto producer() {
  while (true)
  {
    std::scoped_lock<std::mutex> lock{m};
    queue.push(std::vector<float>(320 * 240, 0));
    cv.notify_one();

    if (quit)
      break;
  }
}

auto main() -> int
{
  auto promise = std::promise<int>{};
  auto future = std::future<int>{};
  return 0;
}

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>


namespace DO::Sara {

  template <class T>
  class SafeQueue
  {
  public:
    inline SafeQueue() = default;

    inline ~SafeQueue() = default;

    inline auto enqueue(T&& t) -> void
    {
      std::lock_guard<std::mutex> lock(m);
      q.push(t);
      c.notify_one();
    }

    inline T dequeue()
    {
      std::unique_lock<std::mutex> lock(m);
      if (q.empty())
        return {};
      T val = q.front();
      q.pop();
      return val;
    }

  private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
  };

}  // namespace DO::Sara

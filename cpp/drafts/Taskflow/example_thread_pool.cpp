#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>


class ThreadPool
{
public:
  ThreadPool(int threads)
  {
    // Create the specified number of threads
    _threads.reserve(threads);
    for (int i = 0; i < threads; ++i)
      _threads.emplace_back(std::bind(&ThreadPool::thread_entry, this, i));
  }

  ~ThreadPool()
  {
    {
      // Unblock any threads and tell them to stop
      std::unique_lock<std::mutex> l(_lock);

      _shutdown = true;
      _cv.notify_all();
    }

    // Wait for all threads to stop
    std::cerr << "Joining threads" << std::endl;
    for (auto& thread : _threads)
      thread.join();
  }

  void run(std::function<void(void)> func)
  {
    // Place a job on the queu and unblock a thread
    std::unique_lock<std::mutex> l(_lock);

    _jobs.emplace(std::move(func));
    _cv.notify_one();
  }

protected:
  void thread_entry(int i)
  {
    std::function<void(void)> job;

    for (;;)
    {
      {
        std::unique_lock<std::mutex> l(_lock);

        while (!_shutdown && _jobs.empty())
          _cv.wait(l);

        if (_jobs.empty())
        {
          // No jobs to do and we are shutting down
          std::cerr << "Thread " << i << " terminates" << std::endl;
          return;
        }

        std::cerr << "Thread " << i << " does a job" << std::endl;
        job = std::move(_jobs.front());
        _jobs.pop();
      }

      // Do the job without holding any locks
      job();
    }
  }

  bool _shutdown = false;
  std::mutex _lock;
  std::condition_variable _cv;
  std::queue<std::function<void(void)>> _jobs;
  std::vector<std::thread> _threads;
};


void silly(int n)
{
  // A silly job for demonstration purposes
  std::cerr << "Sleeping for " << n << " seconds" << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(n));
}


int main()
{
  // Create two threads
  ThreadPool p{2};

  // Assign them 4 jobs
  p.run(std::bind(silly, 1));
  p.run(std::bind(silly, 2));
  p.run(std::bind(silly, 3));
  p.run(std::bind(silly, 4));
}

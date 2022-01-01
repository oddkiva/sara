#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <taskflow/taskflow.hpp>


namespace sara = DO::Sara;


template <class T>
class SafeQueue
{
public:
  SafeQueue(void)
    : q()
    , m()
    , c()
  {
  }

  ~SafeQueue(void)
  {
  }

  void enqueue(T t)
  {
    std::lock_guard<std::mutex> lock(m);
    q.push(t);
    c.notify_one();
  }

  T dequeue(void)
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

template<typename T = float>
struct DisplayTask
{
  sara::Image<T> image;
  int index = -1;

  inline DisplayTask() = default;

  inline DisplayTask(sara::Image<T> im, int id)
    : image{std::move(im)}
    , index{id}
  {
  }

  inline DisplayTask(const DisplayTask& task) = default;

  inline DisplayTask(DisplayTask&& task)
    : image{std::move(task.image)}
    , index{task.index}
  {
  }

  inline ~DisplayTask() = default;

  inline auto run() -> void
  {
    if (index == -1 || image.data() == nullptr)
      return;
    auto image_rgb = image.template convert<sara::Rgb8>();
    sara::draw_text(image_rgb, 100, 50, std::to_string(index), sara::White8, 30);
    sara::display(image_rgb);
    std::cout << "Showing frame " << index << std::endl;
  }
};


GRAPHICS_MAIN()
{
  const auto video_path =
      "/Users/david/Desktop/Datasets/brockwell-park-varying-focal-length.mov";
  auto video_stream = sara::VideoStream{video_path};
  auto video_frame = video_stream.frame();
  auto video_frame_gray = sara::Image<float>{video_stream.sizes()};
  auto current_frame = std::atomic_int32_t{-1};
  auto last_frame_shown = std::atomic_int32_t{-1};
  auto video_stream_end = false;

  auto display_queue = SafeQueue<DisplayTask<float>>{};
  auto display_async_task = std::thread{
    [&display_queue, &last_frame_shown, &current_frame, &video_stream_end] {
      while (!video_stream_end)
      {
        auto task = display_queue.dequeue();
        if (task.index < last_frame_shown || task.index + 3 < current_frame)
           continue;
        last_frame_shown = task.index;
        task.run();
      }
    }
  };

  sara::create_window(video_stream.sizes());

  auto taskflow = tf::Taskflow{};
  auto executor = tf::Executor{};

  auto read_video_frame =
      taskflow
          .emplace([&video_stream, &current_frame, &video_stream_end]() {
            video_stream_end = !video_stream.read();
            if (!video_stream_end)
            {
              ++current_frame;
              std::cout << "Read frame " << current_frame << std::endl;
            }
            else
              std::cout << "Finished reading video stream" << std::endl;
          })
          .name("Read frame");

  auto color_convert =
      taskflow
          .emplace([&video_frame, &video_frame_gray]() {
            sara::from_rgb8_to_gray32f(video_frame, video_frame_gray);
          })
          .name("To grayscale");

  auto display = taskflow
                     .emplace([&display_queue, &video_frame_gray, &current_frame] {
                       display_queue.enqueue({video_frame_gray, current_frame});
                     })
                     .name("display");

  read_video_frame.precede(color_convert);
  color_convert.precede(display);

  executor.run_until(taskflow,
                     [&video_stream_end]() { return video_stream_end; }).wait();
  display_async_task.join();

  return 0;
}

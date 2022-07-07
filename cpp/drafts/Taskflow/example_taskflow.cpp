#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing/FastColorConversion.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <drafts/Taskflow/DisplayTask.hpp>
#include <drafts/Taskflow/SafeQueue.hpp>

#include <taskflow/taskflow.hpp>


namespace sara = DO::Sara;


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
#ifdef _WIN32
  const auto video_path = sara::select_video_file_from_dialog_box();
  if (video_path.empty())
    return 1;
#else
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << "  VIDEO_FILEPATH" << std::endl;
    return 1;
  }
  const auto video_path = std::string{argv[1]};
#endif
  auto video_stream = sara::VideoStream{video_path};
  auto video_frame = video_stream.frame();
  auto video_frame_gray = sara::Image<float>{video_stream.sizes()};
  auto current_frame = std::atomic_int32_t{-1};
  auto last_frame_shown = std::atomic_int32_t{-1};
  auto video_stream_end = false;

  auto display_queue = sara::SafeQueue<sara::DisplayTask<float>>{};
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
      }};

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

  auto display =  //
      taskflow
          .emplace([&display_queue, &video_frame_gray, &current_frame] {
            display_queue.enqueue({video_frame_gray, current_frame});
          })
          .name("display");

  read_video_frame.precede(color_convert);
  color_convert.precede(display);

  executor
      .run_until(taskflow, [&video_stream_end]() { return video_stream_end; })
      .wait();
  display_async_task.join();

  return 0;
}

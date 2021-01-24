#define BOOST_TEST_MODULE "VideoIO/VideoWriter Class"

#include <DO/Sara/Core.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <boost/test/unit_test.hpp>

#include <filesystem>

using namespace DO::Sara;


BOOST_AUTO_TEST_SUITE(TestVideoWriter)

BOOST_AUTO_TEST_CASE(test_video_writer)
{
  const auto filepath =
      (std::filesystem::temp_directory_path() / "test.mp4")  //
          .string();

  // TODO: test it more thoroughly later.
  {
    // Dummy.
    auto image = Image<Rgb8>{320, 240};
    image.flat_array().fill(Red8);

    VideoWriter video_writer{filepath, image.sizes()};
    // Dummy write.
    for (auto i = 0; i < 25; ++i)
      video_writer.write(image);
  }
  std::filesystem::remove(filepath);

  // Test the creation of a dummy audio-video file.
  {
    VideoWriter video_writer{filepath, {320, 240}};
    video_writer.generate_dummy();
  }
  std::filesystem::remove(filepath);
}

BOOST_AUTO_TEST_SUITE_END()

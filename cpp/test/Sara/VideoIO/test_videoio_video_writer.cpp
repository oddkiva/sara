#define BOOST_TEST_MODULE "VideoIO/VideoWriter Class"

#include <DO/Sara/Core.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


using namespace DO::Sara;

namespace fs = boost::filesystem;


BOOST_AUTO_TEST_SUITE(TestVideoWriter)

BOOST_AUTO_TEST_CASE(test_video_writer)
{
  const auto filepath = (fs::temp_directory_path() / "test.mp4")  //
                            .string();

  {
    // Dummy image.
    auto image = Image<Rgb8>{320, 240};
    image.flat_array().fill(Red8);

    VideoWriter video_writer{filepath, image.sizes(), 25};
    // Dummy write.
    for (auto i = 0; i < 25; ++i)
      video_writer.write(image);
    video_writer.finish();

    for (auto iter = 0; iter < 3; ++iter)
    {
      VideoStream video_stream{filepath};
      BOOST_CHECK_EQUAL(video_stream.sizes(), image.sizes());
      for (auto i = 10; i < 15; ++i)
      {
        BOOST_CHECK(video_stream.read());
        for (auto p = video_stream.frame().begin();
             p != video_stream.frame().end(); ++p)
          BOOST_REQUIRE_LE(
              (p->cast<int>() - Red8.cast<int>()).lpNorm<Eigen::Infinity>(), 3);
      }
    }

    VideoStream video_stream{filepath};
    BOOST_CHECK_EQUAL(video_stream.sizes(), image.sizes());
    BOOST_CHECK(video_stream.read());
  }
  fs::remove(filepath);
}

BOOST_AUTO_TEST_SUITE_END()

#define BOOST_TEST_MODULE "VideoIO/VideoStream Class"

#include <boost/test/unit_test.hpp>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace DO::Sara;


class TestFixtureForVideoIO
{
protected:
  std::string video_filename;
  VideoStream video_stream;

public:
  TestFixtureForVideoIO()
  {
    video_filename = src_path("hale_bopp_1.mpg");
  }
};


BOOST_FIXTURE_TEST_SUITE(TestVideoIO, TestFixtureForVideoIO)

BOOST_AUTO_TEST_CASE(test_empty_constructor)
{
  VideoStream video_stream;
}

BOOST_AUTO_TEST_CASE(test_read_valid_video)
{
  VideoStream video_stream{video_filename};
  BOOST_CHECK_EQUAL(320, video_stream.width());
  BOOST_CHECK_EQUAL(240, video_stream.height());
}

BOOST_AUTO_TEST_CASE(test_read_video_with_wrong_filepath)
{
  BOOST_CHECK_THROW(VideoStream _video_stream("orio_1.mpg"),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(test_read_valid_video_with_invalid_image_frame)
{
  VideoStream video_stream{video_filename};
  auto frame = Image<Rgb8>{};
  BOOST_CHECK_THROW(video_stream >> frame, std::domain_error);
}

BOOST_AUTO_TEST_CASE(test_read_frames_sequentially)
{
  VideoStream video_stream{video_filename};
  auto frame = Image<Rgb8>{video_stream.sizes()};

  for (auto i = 0; i < 3; ++i)
    video_stream >> frame;
}

BOOST_AUTO_TEST_CASE(test_seek_frame)
{
  VideoStream video_stream{video_filename};
  auto frame = Image<Rgb8>{video_stream.sizes()};

  for (auto i = 0; i < 5; ++i)
    video_stream >> frame;

  VideoStream video_stream2{video_filename};
  auto frame2 = Image<Rgb8>{video_stream.sizes()};

  video_stream.seek(4);
  video_stream2 >> frame2;

  for (auto p = frame.begin(), p2 = frame2.begin(); p != frame.end(); ++p, ++p2)
    BOOST_REQUIRE_EQUAL(*p, *p2);
}

BOOST_AUTO_TEST_SUITE_END()

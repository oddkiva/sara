#include <gtest/gtest.h>

#include <DO/Core.hpp>
#include <DO/VideoIO.hpp>


#ifdef _WIN32
# include <vld.h>
#endif


using namespace DO;


class TestVideoIO : public testing::Test
{
protected:
  std::string video_filename;
  VideoStream video_stream;

  TestVideoIO() : testing::Test()
  {
    video_filename = src_path("hale_bopp_1.mpg");
  }
};


TEST_F(TestVideoIO, test_empty_constructor)
{
  VideoStream video_stream;
}

TEST_F(TestVideoIO, test_read_valid_video)
{
  VideoStream video_stream(video_filename);
}

TEST_F(TestVideoIO, test_read_video_with_wrong_filepath)
{
  EXPECT_THROW(VideoStream _video_stream("orio_1.mpg"),
               std::runtime_error);
}

TEST_F(TestVideoIO, test_read_frames_sequentially)
{
  VideoStream video_stream(video_filename);
  Image<Rgb8> frame;

  for (int i = 0; i < 3; ++i)
    video_stream >> frame;
}

TEST_F(TestVideoIO, test_seek_frame)
{
  VideoStream video_stream(video_filename);
  Image<Rgb8> frame;

  for (int i = 0; i < 5; ++i)
    video_stream >> frame;

  VideoStream video_stream2(video_filename);
  Image<Rgb8> frame2;

  video_stream.seek(4);
  video_stream2.read(frame2);

  for (auto p = frame.begin(), p2 = frame2.begin(); p != frame.end();
       ++p, ++p2)
    ASSERT_EQ(*p, *p2);
}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
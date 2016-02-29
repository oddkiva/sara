#include <gtest/gtest.h>

#include <DO/Sara/Core.hpp>
#include <DO/Sara/VideoIO.hpp>


using namespace DO::Sara;


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
  VideoStream video_stream{ video_filename };
  EXPECT_EQ(320, video_stream.width());
  EXPECT_EQ(240, video_stream.height());
}

TEST_F(TestVideoIO, test_read_video_with_wrong_filepath)
{
  EXPECT_THROW(VideoStream _video_stream("orio_1.mpg"),
               std::runtime_error);
}

TEST_F(TestVideoIO, test_read_valid_video_with_invalid_image_frame)
{
  VideoStream video_stream{ video_filename };
  auto frame = Image<Rgb8>{};
  EXPECT_THROW(video_stream.read(frame), std::domain_error);
}

TEST_F(TestVideoIO, test_read_frames_sequentially)
{
  VideoStream video_stream{ video_filename };
  auto frame = Image<Rgb8>{ video_stream.sizes() };

  for (auto i = 0; i < 3; ++i)
    video_stream >> frame;
}

TEST_F(TestVideoIO, test_seek_frame)
{
  VideoStream video_stream{ video_filename };
  auto frame = Image<Rgb8>{ video_stream.sizes() };

  for (auto i = 0; i < 5; ++i)
    video_stream >> frame;

  VideoStream video_stream2{ video_filename };
  auto frame2 = Image<Rgb8>{ video_stream.sizes() };

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

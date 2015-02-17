#include <gtest/gtest.h>

#include <DO/Core.hpp>
#include <DO/Graphics.hpp>

extern "C" {
# include <libavcodec/avcodec.h>
# include <libavformat/avformat.h>
# include <libavformat/avio.h>
# include <libavutil/file.h>
}

#ifdef _WIN32
# include <vld.h>
#endif

using namespace std;
using namespace DO;


inline
Yuv8 get_yuv_pixel(const AVFrame *frame, int x, int y) 
{
  Yuv8 yuv;
  yuv(0) = frame->data[0][y*frame->linesize[0] + x];
  yuv(1) = frame->data[1][y/2*frame->linesize[1] + x/2];
  yuv(2) = frame->data[2][y/2*frame->linesize[2] + x/2];
  return yuv;
}

inline
unsigned char clamp(int value)
{
  if (value < 0)
    return 0;
  if (value > 255)
    return 255;
  return value;
}

Rgb8 convert(const Yuv8& yuv)
{
  Rgb8 rgb;
  int C = yuv(0) - 16;
  int D = yuv(1) - 128;
  int E = yuv(2) - 128;
  rgb(0) = clamp((298*C + 409*E + 128) >> 8);
  rgb(1) = clamp((298*C - 100*D - 208*E + 128) >> 8);
  rgb(2) = clamp((298*C + 516*D + 128) >> 8);
  return rgb;
}


namespace DO {

  class VideoStream : public std::streambuf
  {
    enum { VIDEO_BUFFER_SIZE = 4096 };

  public:
    VideoStream()
    {
      if (!_registered_all_codecs)
      {
        av_register_all();
        _registered_all_codecs = true;
      }
      init();
    }

    VideoStream(const std::string& file_path)
      : VideoStream()
    {
      open(file_path);
    }

    ~VideoStream()
    {
      close();
    }

    void open(const std::string& file_path)
    {
      _video_file = fopen(file_path.c_str(), "rb");
      if (!_video_file)
        throw std::runtime_error("Could not open file!");

      _video_frame = av_frame_alloc();
      if (!_video_frame)
        throw std::runtime_error("Could not allocate video frame");

      _video_frame_pos = 0;
    }

    void close()
    {
      if (_video_codec_context)
      {
        avcodec_close(_video_codec_context);
        avcodec_free_context(&_video_codec_context);
        _video_codec_context = nullptr;
        _video_codec = nullptr;
      }

      if (_video_frame)
      {
        av_frame_free(&_video_frame);
        _video_frame = nullptr;
        _video_frame_pos = std::numeric_limits<size_t>::max();
      }

      if (_video_file)
      {
        fclose(_video_file);
        _video_file = nullptr;
      }
    }

    void seek(std::size_t frame_pos)
    {
    }

    bool decode(Image<Rgb8>& video_frame)
    {
      int len, got_video_frame;
      len = avcodec_decode_video2(_video_codec_context, _video_frame, &got_video_frame, &_video_packet);
      if (len < 0)
      {
        fprintf(stderr, "Error while decoding frame %d\n", _video_frame_pos);
        return false;
      }

      if (got_video_frame)
      {
        int w = _video_codec_context->width;
        int h = _video_codec_context->height;

        if (video_frame.width() != w || video_frame.height() != h)
          video_frame.resize(w, h);

        for (int y = 0; y < h; ++y)
        {
          for (int x = 0; x < w; ++x)
          {
            Yuv8 yuv = get_yuv_pixel(_video_frame, x, y);
            video_frame(x, y) = ::convert(yuv);
          }
        }

        ++_video_frame_pos;
      }
      if (_video_packet.data) {
        _video_packet.size -= len;
        _video_packet.data += len;
      }

      return true;
    }

    bool read(Image<Rgb8>& video_frame)
    {
      _video_packet.size = fread(_video_buffer, 1, VIDEO_BUFFER_SIZE, _video_file);
      if (_video_packet.size == 0)
        return false;

      _video_packet.data = _video_buffer;
      while (_video_packet.size > 0)
        if (!decode(video_frame))
          return false;

      return true;
    }

    friend inline VideoStream& operator>>(VideoStream& video_stream, Image<Rgb8>& video_frame)
    {
      if (!video_stream.read(video_frame))
        throw runtime_error("Could not read video frame");
      return video_stream;
    }

private:
    void init()
    {
      // Set end of buffer to 0 (this ensures that no over-reading happens for
      // damaged MPEG streams).
      memset(_video_buffer + VIDEO_BUFFER_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);

      // Initialize the video packet.
      av_init_packet(&_video_packet);

      // 1. Try to find video codec.
      _video_codec = avcodec_find_decoder(AV_CODEC_ID_MPEG1VIDEO);
      if (!_video_codec)
        throw std::runtime_error("Could not find video codec!");

      // 2. Allocate video codec context.
      _video_codec_context = avcodec_alloc_context3(_video_codec);
      if (!_video_codec_context)
        throw std::runtime_error("Could not allocate video codec context!");

      // 3. Set the settings for the video codec context.
      if (_video_codec->capabilities & CODEC_CAP_TRUNCATED)
        _video_codec_context->flags|= CODEC_FLAG_TRUNCATED; /* we do not send complete frames */

      // 4. Open the video codec context.
      if (avcodec_open2(_video_codec_context, _video_codec, NULL) < 0)
        throw std::runtime_error("Could not open video codec context!");
    }

  private:
    static bool _registered_all_codecs;

    FILE *_video_file = nullptr;
    AVCodec *_video_codec = nullptr;
    AVCodecContext *_video_codec_context = nullptr;
    AVFrame *_video_frame = nullptr;
    size_t _video_frame_pos = std::numeric_limits<size_t>::max();
    AVPacket _video_packet;
    uint8_t _video_buffer[VIDEO_BUFFER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];
  };

  bool VideoStream::_registered_all_codecs = false;

}


class TestVideoIO : public testing::Test
{
protected:
  std::string video_filename;
  VideoStream video_stream;

  TestVideoIO() : testing::Test()
  {
    video_filename = srcPath("orion_1.mpg");
    cout << video_filename << endl;
  }
};


TEST_F(TestVideoIO, test_constructor)
{
  VideoStream video_stream;
}

TEST_F(TestVideoIO, test_read_valid_video)
{
  VideoStream video_stream(video_filename);
}

TEST_F(TestVideoIO, test_read_video_with_wrong_filepath)
{
  EXPECT_THROW(VideoStream video_stream("orio_1.mpg"),
               std::runtime_error);
}

TEST_F(TestVideoIO, test_read_frames_sequentially)
{
  VideoStream video_stream(video_filename);
  VideoStream video_stream2(video_filename);
  Image<Rgb8> frame;

  for (int i = 0; i < 5; ++i)
  {
    video_stream >> frame;
    video_stream2 >> frame;
  }
}

TEST_F(TestVideoIO, test_seek_frame)
{
  VideoStream video_stream(video_filename);

  size_t frame_index = 5;
  video_stream.seek(frame_index);

  Image<Rgb8> frame;
  //video_stream >> frame;
}


//#define TEST_FEATURES
#ifdef TEST_FEATURES
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#else
GRAPHICS_MAIN_SIMPLE()
{
  const string video_filepath = srcPath("orion_1.mpg");
  //const string video_filepath = "C:/Users/David/Desktop/STREAM/00102.MTS";

  VideoStream video_stream(video_filepath);
  Image<Rgb8> video_frame;

  while (true)
  {
    if (!video_stream.read(video_frame))
      break;
    //video_stream >> video_frame;
    if (!active_window())
      create_window(video_frame.sizes());
    display(video_frame);
  }

  return 0;
}
#endif

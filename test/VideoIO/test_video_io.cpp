#include <gtest/gtest.h>

#include <DO/Core.hpp>
#include <DO/Graphics.hpp>

extern "C" {
# include <libavcodec/avcodec.h>
# include <libavformat/avformat.h>
# include <libavformat/avio.h>
# include <libavutil/file.h>
}


using namespace std;
using namespace DO;


const size_t INBUF_SIZE = 4096;

Yuv8 yuv_pixel(AVFrame *frame, int x, int y) 
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

// Thanks to Wikipedia!
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


static int decode_write_frame(Image<Rgb8>& image,
                              AVCodecContext *avctx,
                              AVFrame *frame,
                              int *frame_count,
                              AVPacket *pkt, int last)
{
  int len, got_frame;
  len = avcodec_decode_video2(avctx, frame, &got_frame, pkt);
  if (len < 0) {
    fprintf(stderr, "Error while decoding frame %d\n", *frame_count);
    return len;
  }

  if (got_frame) {
    //printf("frame %d\n", *frame_count);
    //fflush(stdout);

    int w = avctx->width;
    int h = avctx->height;
    if (!getActiveWindow())
      openWindow(w, h);

    if (image.width() != w || image.height())
      image.resize(w, h);

    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
      {
        Yuv8 yuv = yuv_pixel(frame, x, y);
        image(x, y) = convert(yuv);
      }

    display(image);
    //milliSleep(20);

    (*frame_count)++;
  }
  if (pkt->data) {
    pkt->size -= len;
    pkt->data += len;
  }
  return 0;
}


static void decode_video(const string& filename)
{
  AVCodec *codec = NULL;
  AVCodecContext *context = NULL;
  int frame_count;
  AVFrame *frame;
  uint8_t inbuf[INBUF_SIZE + FF_INPUT_BUFFER_PADDING_SIZE];
  AVPacket packet;
  av_init_packet(&packet);

  // Set end of buffer to 0 (this ensures that no over-reading happens for
  // damaged mpeg streams).
  memset(inbuf + INBUF_SIZE, 0, FF_INPUT_BUFFER_PADDING_SIZE);

  /* find the mpeg1 video decoder */
  codec = avcodec_find_decoder(AV_CODEC_ID_MPEG1VIDEO);
  if (!codec) {
    fprintf(stderr, "Codec not found\n");
    exit(1);
  }
  context = avcodec_alloc_context3(codec);
  if (!context)
  {
    fprintf(stderr, "Could not allocate video codec context\n");
    exit(1);
  }
  if (codec->capabilities & CODEC_CAP_TRUNCATED)
    context->flags|= CODEC_FLAG_TRUNCATED; /* we do not send complete frames */

  if (avcodec_open2(context, codec, NULL) < 0) {
    fprintf(stderr, "Could not open codec\n");
    exit(1);
  }

  // Read video file.
  FILE *file = fopen(filename.c_str(), "rb");
  if (!file)
  {
    fprintf(stderr, "Could not open %s\n", filename);
    exit(1);
  }

  // Allocate frame.
  frame = av_frame_alloc();
  if (!frame) {
    fprintf(stderr, "Could not allocate video frame\n");
    exit(1);
  }

  Image<Rgb8> image;

  // Read frame by frame.
  frame_count = 0;
  for ( ; ; )
  {
    packet.size = fread(inbuf, 1, INBUF_SIZE, file);
    if (packet.size == 0)
      break;

    packet.data = inbuf;
    while (packet.size > 0)
      if (decode_write_frame(image, context, frame, &frame_count, &packet, 0) < 0)
        exit(1);
  }

  packet.data = NULL;
  packet.size = 0;

  fclose(file);
  avcodec_close(context);
  av_free(context);
  av_frame_free(&frame);
  printf("\n");
}


namespace DO { namespace Detail {

  class Codec
  {
  public:
    Codec(AVCodecID codec_id = AV_CODEC_ID_MPEG1VIDEO)
    {
      if (_registered_all_codecs)
        av_register_all();

      /* find the mpeg1 video decoder */
      _codec = avcodec_find_decoder(codec_id);
      if (!_codec)
        throw std::runtime_error("Cannot find codec");
    }

    operator AVCodec *() const
    {
      return _codec;
    }

  private:
    AVCodec *_codec;
    static bool _registered_all_codecs;
  };
  bool Codec::_registered_all_codecs = false;

  class CodecContext 
  {
  public:
    CodecContext(const Codec& codec)
      : _context(nullptr)
    {
      _context = avcodec_alloc_context3(codec);
      if (!_context)
        throw runtime_error("Could not allocate video codec context");
    }

    ~CodecContext()
    {
      _context = avcodec_free_context(_context)
    }

  private:
    AVCodecContext *_context;
  };


} /* namespace Detail */
} /* namespace DO */


namespace DO
{
  class VideoStream
  {
  public:
    VideoStream()
    {
    }

    VideoStream(const string& filename)
    {
    }

    void seek(std::size_t frame_index)
    {
    }

    operator bool() const
    {
      return true;
    }

  private:
    size_t _current_index;
  };

  template <typename T>
  VideoStream& operator>>(VideoStream& in_video, Image<T>& frame)
  {
    return in_video;
  }

}


class TestVideoIO : public testing::Test
{
protected:
  string video_filename;
  VideoStream video_stream;

  TestVideoIO() : testing::Test()
  {
    video_filename = srcPath("orion_1.mpg");
    cout << video_filename << endl;
  }

};


TEST_F(TestVideoIO, test_read_video)
{
  VideoStream video_stream(video_filename);
  EXPECT_TRUE(video_stream);
}


TEST_F(TestVideoIO, test_read_frames_sequentially)
{
  VideoStream video_stream(video_filename);
  EXPECT_TRUE(video_stream);

  Image<Rgb8> frame;

  for (int i = 0; i < 5; ++i)
    video_stream >> frame;
}


TEST_F(TestVideoIO, test_seek_frame)
{
  VideoStream video_stream(video_filename);
  EXPECT_TRUE(video_stream);

  size_t frame_index = 5;
  video_stream.seek(frame_index);

  Image<Rgb8> frame;
  video_stream >> frame;

}


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


//GRAPHICS_MAIN_SIMPLE()
//{
//  /* Register all the codecs. */
//  avcodec_register_all();
//
//
//  const string filename = srcPath("orion_1.mpg");
//  decode_video(filename);
//  return 0;
//}
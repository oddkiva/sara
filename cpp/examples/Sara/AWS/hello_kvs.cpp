#include <DO/Sara/Core.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/VideoIO.hpp>

#include <aws/core/Aws.h>
#include <aws/ec2/EC2Client.h>
#include <aws/ec2/model/DescribeInstancesRequest.h>
#include <aws/kinesis-video-archived-media/KinesisVideoArchivedMediaClient.h>
#include <aws/kinesisvideo/KinesisVideoClient.h>
#include <aws/kinesisvideo/model/GetDataEndpointRequest.h>

#include <cstdlib>
#include <iostream>


namespace sara = DO::Sara;
namespace kv = Aws::KinesisVideo;
namespace kvam = Aws::KinesisVideoArchivedMedia;


static constexpr auto KVS_STREAM_NAME = "oddkiva-test-video-stream";


class AwsContext
{
public:
  AwsContext(const Aws::SDKOptions& options)
    : _options{options}
  {
    std::cout << "Init AWS context..." << std::endl;
    Aws::InitAPI(_options);
  }

  ~AwsContext()
  {
    std::cout << "Deinit AWS context..." << std::endl;
    Aws::ShutdownAPI(_options);
  }

  const Aws::SDKOptions _options;
};


GRAPHICS_MAIN()
{
  auto options = Aws::SDKOptions{};
  options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Trace;
  const auto aws_context = AwsContext(options);

  auto client_config = Aws::Client::ClientConfiguration{};
  client_config.region = "eu-west-2";

  const auto kv_client = kv::KinesisVideoClient{client_config};
  const auto kv_data_endpoint_outcome = kv_client.GetDataEndpoint(
      kv::Model::GetDataEndpointRequest{}
          .WithAPIName(kv::Model::APIName::GET_HLS_STREAMING_SESSION_URL)
          .WithStreamName(KVS_STREAM_NAME));
  if (!kv_data_endpoint_outcome.IsSuccess())
  {
    std::cerr << "Failed to kinesis video data endpoint: "
              << kv_data_endpoint_outcome.GetError() << std::endl;
    return EXIT_FAILURE;
  }
  const auto kv_data_endpoint =
      kv_data_endpoint_outcome.GetResult().GetDataEndpoint();
  std::cout << "[Kinesis-Video] " << kv_data_endpoint << std::endl;

  auto kvam_config =
      kvam::KinesisVideoArchivedMediaClientConfiguration{client_config};
  kvam_config.endpointOverride = kv_data_endpoint;
  const auto kvam_client = kvam::KinesisVideoArchivedMediaClient{kvam_config};

  const auto kvam_streaming_session_url_outcome =
      kvam_client.GetHLSStreamingSessionURL(
          kvam::Model::GetHLSStreamingSessionURLRequest{}
              .WithStreamName(KVS_STREAM_NAME)
              .WithPlaybackMode(kvam::Model::HLSPlaybackMode::LIVE)
              .WithMaxMediaPlaylistFragmentResults(25ull)
              .WithExpires(43200));
  if (!kvam_streaming_session_url_outcome.IsSuccess())
  {
    std::cerr << "Failed to kinesis video archived media streaming URL"
              << std::endl;
    return EXIT_FAILURE;
  }
  const auto kvam_streaming_session_url =
      kvam_streaming_session_url_outcome.GetResult()
          .GetHLSStreamingSessionURL();
  if (kvam_streaming_session_url.empty())
  {
    std::cerr << "Invalid video archived media streaming URL" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "[Kinesis-Video-ArchivedMedia] " << kvam_streaming_session_url
            << std::endl;

  auto video_stream = sara::VideoStream{kvam_streaming_session_url};
  const auto video_frame = video_stream.frame();

  SARA_DEBUG << "Frame rate = " << video_stream.frame_rate() << std::endl;
  SARA_DEBUG << "Frame sizes = " << video_stream.sizes().transpose()
             << std::endl;
  SARA_DEBUG << "Frame rotation angle = " << video_stream.rotation_angle()
             << std::endl;

  sara::create_window(video_stream.sizes());
  while (true)
  {
    sara::tic();
    auto has_frame = video_stream.read();
    sara::toc("Read frame");

    if (!has_frame)
      break;

    sara::tic();
    sara::display(video_frame);
    sara::toc("Display");
  }

  sara::close_window();

  return EXIT_SUCCESS;
}

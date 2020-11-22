//! @example

#include <memory>
#include <vector>

#include <lqt/lqt.h>

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <DO/Shakti/ImageProcessing.hpp>
#include <DO/Shakti/MultiArray.hpp>
#include <DO/Shakti/Segmentation.hpp>
#include <DO/Shakti/Utilities.hpp>


namespace DO { namespace Sara {

  class QuicktimeVideoStream
  {
  public:
    QuicktimeVideoStream() = default;

    QuicktimeVideoStream(const std::string& video_filepath)
    {
      _video_file = quicktime_open(video_filepath.c_str(), 1, 0);
      _num_tracks = quicktime_video_tracks(_video_file);

      _current_track = 0;
      _current_track_supported = false;
    }

    ~QuicktimeVideoStream()
    {
      quicktime_close(_video_file);
    }

    Vector2i sizes() const
    {
      return Vector2i{quicktime_video_width(_video_file, _current_track),
                      quicktime_video_height(_video_file, _current_track)};
    }

    int depth() const
    {
      return quicktime_video_depth(_video_file, _current_track);
    }

    std::string color_model() const
    {
      auto color_model = lqt_get_cmodel(_video_file, _current_track);
      auto color_model_name = lqt_colormodel_to_string(color_model);
      return color_model_name;
    }

    int current_timestamp() const
    {
      return lqt_frame_time(_video_file, _current_track);
    }

    double current_frame_rate() const
    {
      return static_cast<double>(
                 lqt_video_time_scale(_video_file, _current_track)) /
             lqt_frame_duration(_video_file, _current_track, nullptr);
    }

    void bind_frame_rows(Image<Rgb8>& frame)
    {
      auto sizes = this->sizes();
      frame.resize(sizes);
      _current_frame_rows.resize(sizes[1]);
      for (int y = 0; y < sizes[1]; ++y)
        _current_frame_rows[y] = reinterpret_cast<unsigned char*>(&frame(0, y));
    }

    bool read(Image<Rgb8>& video_frame, bool bind_frame = true)
    {
      if (!_current_track_supported &&
          quicktime_supported_video(_video_file, _current_track) == 0)
      {
        std::cout << "Movie track " << _current_track
                  << " is unsupported by liquicktime!" << std::endl;
        return false;
      }

      if (video_frame.sizes() != sizes() || bind_frame)
        bind_frame_rows(video_frame);
      quicktime_decode_video(_video_file, _current_frame_rows.data(),
                             _current_track);

      return true;
    }

  private:
    quicktime_t* _video_file;
    int _num_tracks;
    std::vector<unsigned char*> _video_rows;

    int _current_track;
    bool _current_track_supported;
    std::vector<unsigned char*> _current_frame_rows;
  };

} /* namespace Sara */
} /* namespace DO */


namespace sara = DO::Sara;
namespace shakti = DO::Shakti;


using namespace std;


template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


void gaussian_demo()
{
  const auto video_filepath =
      string{"/home/david/Desktop/HAVAS_DANONE_PITCH_EP_1011.mov"};
  auto video_stream = sara::QuicktimeVideoStream{video_filepath};
  auto video_frame = sara::Image<sara::Rgb8>{};

  auto in_frame = sara::Image<float>{};
  auto out_frame = sara::Image<float>{};

  const auto sigma = 3.f;
  auto apply_gaussian_filter = shakti::GaussianFilter{sigma};

  video_stream.bind_frame_rows(video_frame);

  int frame = 0;
  while (true)
  {
    cout << endl << "[Read Frame] frame = " << frame << endl;
    video_stream.read(video_frame, false);

    shakti::tic();
    in_frame = video_frame.convert<float>();
    out_frame.resize(in_frame.sizes());
    shakti::toc("Color conversion");

    shakti::tic();
    apply_gaussian_filter(out_frame.data(), in_frame.data(),
                          in_frame.sizes().data());
    shakti::toc("GPU Gaussian");

    shakti::tic();
    shakti::compute_laplacian(out_frame.data(), out_frame.data(),
                              out_frame.sizes().data());
    shakti::toc("GPU Laplacian");

    if (!sara::active_window())
      sara::create_window(video_frame.sizes());
    sara::display(color_rescale(out_frame));

    ++frame;
  }
}


void superpixel_demo()
{
  using namespace sara;

  auto devices = shakti::get_devices();
  devices.front().make_current_device();
  cout << devices.front() << endl;

  const auto video_filepath =
      string{"/home/david/Desktop/HAVAS_DANONE_PITCH_EP_1011.mov"};
  auto video_stream = sara::QuicktimeVideoStream{video_filepath};
  auto video_frame = sara::Image<sara::Rgb8>{};
  auto video_frame_index = int{0};

  video_stream.bind_frame_rows(video_frame);

  shakti::SegmentationSLIC slic;
  slic.set_distance_weight(1e-4f);

  auto frame_rgba32f = Image<Rgba32f>{};
  auto labels = Image<int>{};
  auto segmentation = Image<Rgba32f>{};
  auto means = vector<Rgba32f>{};
  auto cardinality = vector<int>{};

  while (true)
  {
    cout << endl << "[Read Frame] frame = " << video_frame_index << endl;
    video_stream.read(video_frame, false);

    if (!active_window())
      create_window(video_frame.sizes());

    frame_rgba32f = video_frame.convert<Rgba32f>();
    labels.resize(video_frame.sizes());
    segmentation.resize(video_frame.sizes());

    Timer t;
    t.restart();
    slic(labels.data(),
         reinterpret_cast<shakti::Vector4f*>(frame_rgba32f.data()),
         frame_rgba32f.sizes().data());
    cout << "Segmentation time = " << t.elapsed_ms() << "ms" << endl;

    means.resize(labels.array().maxCoeff() + 1);
    cardinality.resize(labels.array().maxCoeff() + 1);
    fill(means.begin(), means.end(), Rgba32f::Zero());
    fill(cardinality.begin(), cardinality.end(), 0);

    // Compute the mean clusters.
    for (int y = 0; y < segmentation.height(); ++y)
      for (int x = 0; x < segmentation.width(); ++x)
      {
        means[labels(x, y)] += frame_rgba32f(x, y);
        ++cardinality[labels(x, y)];
      }
    for (size_t i = 0; i < means.size(); ++i)
      means[i] /= cardinality[i];

    // Update the segmentation.
    for (int y = 0; y < segmentation.height(); ++y)
      for (int x = 0; x < segmentation.width(); ++x)
        segmentation(x, y) = means[labels(x, y)];

    display(segmentation);

    ++video_frame_index;
    cout << endl;
  }
}


GRAPHICS_MAIN()
{
  superpixel_demo();
  return 0;
}

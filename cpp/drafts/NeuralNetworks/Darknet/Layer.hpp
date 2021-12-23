#pragma once

#include <DO/Sara/Core/Tensor.hpp>

#include <boost/algorithm/string.hpp>


namespace DO::Sara::Darknet {

  struct Layer
  {
    virtual ~Layer() = default;

    virtual auto parse_line(const std::string&) -> void = 0;

    virtual auto to_output_stream(std::ostream& os) const -> void = 0;

    virtual auto forward(const TensorView_<float, 4>&)
        -> const TensorView_<float, 4>&
    {
      throw std::runtime_error{"Unimplemented!"};
      return output;
    }

    friend inline auto operator<<(std::ostream& os, const Layer& l)
        -> std::ostream&
    {
      l.to_output_stream(os);
      return os;
    }

    bool debug = false;
    std::string type;
    Eigen::Vector4i input_sizes = Eigen::Vector4i::Constant(-1);
    Eigen::Vector4i output_sizes = Eigen::Vector4i::Constant(-1);

    Tensor_<float, 4> output;
  };

  struct Input : Layer
  {
    inline auto update_output_sizes(bool inference = true) -> void
    {
      if (inference)
        batch() = 1;
      output_sizes(1) = 3;
      output.resize(output_sizes);
    }

    inline auto parse_line(const std::string& line) -> void override
    {
      auto line_split = std::vector<std::string>{};
      boost::split(line_split, line, boost::is_any_of("="),
                   boost::token_compress_on);
      for (auto& str : line_split)
        boost::trim(str);

      const auto& key = line_split[0];
      if (key == "width")
        width() = std::stoi(line_split[1]);
      else if (key == "height")
        height() = std::stoi(line_split[1]);
      else if (key == "batch")
        batch() = std::stoi(line_split[1]);
    }

    inline auto to_output_stream(std::ostream& os) const -> void override
    {
      os << "- input width  = " << width() << "\n";
      os << "- input height = " << height() << "\n";
      os << "- input batch  = " << batch() << "\n";
    }

    inline auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override
    {
      output = x;
      return output;
    }

    inline auto width() noexcept -> int&
    {
      return output_sizes(3);
    };

    inline auto height() noexcept -> int&
    {
      return output_sizes(2);
    }

    inline auto batch() noexcept -> int&
    {
      return output_sizes(0);
    }

    inline auto width() const noexcept -> const int&
    {
      return output_sizes(3);
    };

    inline auto height() const -> const int&
    {
      return output_sizes(2);
    }

    inline auto batch() const -> const int&
    {
      return output_sizes(0);
    }
  };

  struct BatchNormalization : Layer
  {
    struct Weights
    {
      Eigen::VectorXf scales;
      Eigen::VectorXf rolling_mean;
      Eigen::VectorXf rolling_variance;
    } weights;

    inline auto resize(const Eigen::Vector4i& sizes)
    {
      const auto& num_filters = sizes(1);
      input_sizes = sizes;
      output_sizes = sizes;
      weights.scales.resize(num_filters);
      weights.rolling_mean.resize(num_filters);
      weights.rolling_variance.resize(num_filters);

      output.resize(sizes);
    }

    inline virtual auto parse_line(const std::string&) -> void
    {
      throw std::runtime_error{"Not Implemented!"};
    }

    inline virtual auto to_output_stream(std::ostream& os) const -> void
    {
      os << "- input          = " << input_sizes.transpose() << "\n";
      os << "- output         = " << output_sizes.transpose() << "\n";
    }

    inline auto load_weights(FILE* fp) -> void
    {
      if (debug)
        std::cout << "Loading BN scales: " << weights.scales.size()
                  << std::endl;
      const auto scales_count = fread(weights.scales.data(), sizeof(float),
                                      weights.scales.size(), fp);
      if (Eigen::Index(scales_count) != weights.scales.size())
        throw std::runtime_error{"Failed to read BN scales!"};

      if (debug)
        std::cout << "Loading BN rolling mean: " << weights.rolling_mean.size()
                  << std::endl;
      const auto rolling_mean_count =
          fread(weights.rolling_mean.data(), sizeof(float),
                weights.rolling_mean.size(), fp);
      if (Eigen::Index(rolling_mean_count) != weights.rolling_mean.size())
        throw std::runtime_error{"Failed to read BN rolling mean!"};

      if (debug)
        std::cout << "Loading BN rolling variance: "
                  << weights.rolling_variance.size() << std::endl;
      const auto rolling_variance_count =
          fread(weights.rolling_variance.data(), sizeof(float),
                weights.rolling_variance.size(), fp);
      if (Eigen::Index(rolling_variance_count) !=
          weights.rolling_variance.size())
        throw std::runtime_error{"Failed to read BN rolling variance!"};
    }
  };

  struct Convolution : Layer
  {
    bool batch_normalize = false;

    int filters;
    int size;
    int stride;
    int pad;
    std::string activation;

    struct Weights
    {
      Tensor_<float, 4> w;
      Eigen::VectorXf b;
    } weights;

    std::unique_ptr<BatchNormalization> bn_layer;

    inline auto update_output_sizes() -> void
    {
      output_sizes = input_sizes;
      output_sizes[1] = filters;
      output_sizes.tail(2) /= stride;

      output.resize(output_sizes);
    }

    inline auto parse_line(const std::string& line) -> void override
    {
      auto line_split = std::vector<std::string>{};
      boost::split(line_split, line, boost::is_any_of("="),
                   boost::token_compress_on);
      for (auto& str : line_split)
        boost::trim(str);

      const auto& key = line_split[0];
      if (key == "batch_normalize")
        batch_normalize = static_cast<bool>(std::stoi(line_split[1]));
      else if (key == "filters")
        filters = std::stoi(line_split[1]);
      else if (key == "size")
        size = std::stoi(line_split[1]);
      else if (key == "stride")
        stride = std::stoi(line_split[1]);
      else if (key == "pad")
        pad = std::stoi(line_split[1]);
      else if (key == "activation")
        activation = line_split[1];
    }

    inline auto to_output_stream(std::ostream& os) const -> void override
    {
      os << "- normalize      = " << batch_normalize << "\n";
      os << "- filters        = " << filters << "\n";
      os << "- size           = " << size << "\n";
      os << "- stride         = " << stride << "\n";
      os << "- pad            = " << pad << "\n";
      os << "- activation     = " << activation << "\n";
      os << "- input          = " << input_sizes.transpose() << "\n";
      os << "- output         = " << output_sizes.transpose();
    }

    inline auto load_weights(FILE* fp, bool inference = true) -> void
    {
      // 1. Read bias weights.
      // 2. Read batch normalization weights.
      // 3. Read convolution weights.
      // 1. Convolutional bias weights.
      weights.b.resize(filters);
      const auto bias_weight_count =
          fread(weights.b.data(), sizeof(float), weights.b.size(), fp);
      if (Eigen::Index(bias_weight_count) != weights.b.size())
        throw std::runtime_error{"Failed to read bias weights!"};
      if (debug)
      {
        std::cout << "Loading Conv B: " << weights.b.size() << std::endl;
        std::cout << weights.b.transpose() << std::endl;
      }

      // 2. Batch normalization weights.
      if (batch_normalize)
      {
        bn_layer = std::make_unique<BatchNormalization>();
        bn_layer->resize(output_sizes);
        bn_layer->load_weights(fp);
      }

      // 3. Convolution kernel weights.
      weights.w.resize({filters, input_sizes(1), size, size});
      if (debug)
        std::cout << "Loading Conv W: " << weights.w.sizes().transpose()
                  << std::endl;
      const auto kernel_weight_count =
          fread(weights.w.data(), sizeof(float), weights.w.size(), fp);
      if (kernel_weight_count != weights.w.size())
        throw std::runtime_error{"Failed to read kernel weights!"};
      if (debug)
      {
        std::cout << "Loading Conv W: " << weights.w.size() << std::endl;
        std::cout << weights.w.vector().transpose().head(10) << std::endl;
        throw 0;
      }

      // Fuse the convolution and the batch normalization in a single
      // operation.
      const auto fuse_conv_bn_layer = inference && batch_normalize;
      if (fuse_conv_bn_layer)
      {
        if (debug)
          std::cout << "Fusing Conv and BN layer" << std::endl;

        static constexpr auto eps = .00001;

        weights.b.array() =
            (weights.b.array().cast<double>() -
             bn_layer->weights.scales.array().cast<double>() *
                 bn_layer->weights.rolling_mean.array().cast<double>() /
                 (bn_layer->weights.rolling_variance.array().cast<double>() +
                  eps)
                     .sqrt())
                .cast<float>();

        for (auto n = 0; n < weights.w.size(0); ++n)
        {
          const auto precomputed =
              bn_layer->weights.scales(n) /
              std::sqrt(double(bn_layer->weights.rolling_variance(n)) + eps);
          weights.w[n].flat_array() *= precomputed;
        }

        bn_layer.reset(nullptr);
      }
    }

    inline virtual auto forward(const TensorView_<float, 4>&x)
        -> const TensorView_<float, 4>& override
    {
      auto& y = output;

      const auto& w = weights.w;
      const auto& b = weights.b;
      const auto offset = -size / 2;

      // Convolve.
      im2col_gemm_convolve(
                           y,
                           x,                                       // the signal
                           w,                                       // the transposed kernel.
                           make_constant_padding(0.f),              // the padding type
                           {x.size(0), x.size(1), stride, stride},  // strides in the convolution
                           {0, 0, offset, offset});                 // offset to center the conv.

      // Bias.
      for (auto n = 0; n < y.size(0); ++n)
        for (auto c = 0; c < y.size(1); ++c)
          y[n][c].flat_array() += b(c);

      if (activation == "leaky")
        y.cwise_transform_inplace([](float& v) { v = v > 0 ? v : 0.1f * v; });
      else if (activation == "linear")
      {
      }
      else
        throw std::runtime_error{"Unsupported activation!"};

      return y;
    }
  };

  //! @brief Concatenates the output of the different layers into a single
  //! output.
  struct Route : Layer
  {
    std::vector<std::int32_t> layers;
    int groups = 1;
    int group_id = -1;

    inline auto
    update_output_sizes(const std::vector<std::unique_ptr<Layer>>& nodes)
        -> void
    {
      // All layers must have the same width, height, and batch size.
      // Only the input channels vary.
      const auto id = layers.front() < 0
                          ? nodes.size() - 1 + layers.front()
                          : layers.front() + 1 /* because of the input layer */;
      input_sizes = nodes[id]->output_sizes;
      output_sizes = nodes[id]->output_sizes;

      auto channels = 0;
      for (const auto& rel_id : layers)
      {
        const auto id = rel_id < 0 ? nodes.size() - 1 + rel_id : rel_id;
        channels += nodes[id]->output_sizes[1];
      }
      output_sizes[1] = channels;

      output_sizes[1] /= groups;

      output.resize(output_sizes);
    }

    inline auto parse_line(const std::string& line) -> void override
    {
      auto line_split = std::vector<std::string>{};
      boost::split(line_split, line, boost::is_any_of("="),
                   boost::token_compress_on);
      for (auto& str : line_split)
        boost::trim(str);

      const auto& key = line_split[0];
      if (key == "groups")
        groups = std::stoi(line_split[1]);
      else if (key == "group_id")
        group_id = std::stoi(line_split[1]);
      else if (key == "layers")
      {
        auto layer_strings = std::vector<std::string>{};
        boost::split(layer_strings, line_split[1], boost::is_any_of(", "),
                     boost::token_compress_on);
        for (const auto& layer_str : layer_strings)
          layers.push_back(std::stoi(layer_str));
      }
    }

    inline auto to_output_stream(std::ostream& os) const -> void override
    {
      os << "- layers         = ";
      for (const auto& layer : layers)
        os << layer << ", ";
      os << "\n";

      os << "- groups         = " << groups << "\n";
      os << "- group_id       = " << group_id << "\n";
      os << "- input          = " << input_sizes.transpose() << "\n";
      os << "- output         = " << output_sizes.transpose();
    }
  };

  struct MaxPool : Layer
  {
    int size = 2;
    int stride = 2;

    inline auto update_output_sizes() -> void
    {
      output_sizes = input_sizes;
      output_sizes.tail(2) /= stride;

      output.resize(output_sizes);
    }

    inline auto parse_line(const std::string& line) -> void override
    {
      auto line_split = std::vector<std::string>{};
      boost::split(line_split, line, boost::is_any_of("="),
                   boost::token_compress_on);
      for (auto& str : line_split)
        boost::trim(str);

      const auto& key = line_split[0];
      if (key == "size")
        size = std::stoi(line_split[1]);
      else if (key == "stride")
        stride = std::stoi(line_split[1]);
    }

    inline auto to_output_stream(std::ostream& os) const -> void override
    {
      os << "- size           = " << size << "\n";
      os << "- stride         = " << stride << "\n";
      os << "- input          = " << input_sizes.transpose() << "\n";
      os << "- output         = " << output_sizes.transpose();
    }

    inline virtual auto forward(const TensorView_<float, 4>&x)
        -> const TensorView_<float, 4>& override
    {
      auto& y = output;
      if (size != 2)
        throw std::runtime_error{
            "MaxPool implementation incomplete! size must be 2"};

      const auto start = Eigen::Vector4i::Zero().eval();
      const auto& end = x.sizes();
      const auto steps = (Eigen::Vector4i{} << 1, 1, stride, stride).finished();

      const auto infx = make_infinite(x, make_constant_padding(0.f));

      auto xi = infx.begin_stepped_subarray(start, end, steps);
      auto yi = y.begin();
      for (; yi != y.end(); ++yi, ++xi)
      {
        const auto& p = xi.position();
        const Matrix<int, 4, 1> s = p;
        const Matrix<int, 4, 1> e = p + Eigen::Vector4i{1, 1, size, size};

        auto x_arr = std::array<float, 4>{};
        auto samples = TensorView_<float, 4>{x_arr.data(), e - s};
        crop(samples, infx, s, e);

        *yi = *std::max_element(samples.begin(), samples.end());
      }

      return y;
    }
  };

  struct Upsample : Layer
  {
    int stride = 2;

    inline auto update_output_sizes() -> void
    {
      output_sizes = input_sizes;
      output_sizes.tail(2) *= stride;
      output.resize(output_sizes);
    }

    inline auto parse_line(const std::string& line) -> void override
    {
      auto line_split = std::vector<std::string>{};
      boost::split(line_split, line, boost::is_any_of("="),
                   boost::token_compress_on);
      for (auto& str : line_split)
        boost::trim(str);

      const auto& key = line_split[0];
      if (key == "stride")
        stride = std::stoi(line_split[1]);
    }

    inline auto to_output_stream(std::ostream& os) const -> void override
    {
      os << "- stride         = " << stride << "\n";
      os << "- input          = " << input_sizes.transpose() << "\n";
      os << "- output         = " << output_sizes.transpose();
    }

    virtual auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override
    {
      auto& y = output;
      for (auto yi = y.begin_array(); yi != y.end(); ++yi)
      {
        Matrix<int, 4, 1> p = yi.position();
        p.tail(2) /= stride;
        *yi = x(p);
      }
      return y;
    }
  };

  struct Yolo : Layer
  {
    //! @brief  The list of anchor box sizes (w[0], h[0]), ..., (w[N-1], h[N-1])
    std::vector<std::int32_t> anchors;
    //! @brief The list of anchor box indices to consider.
    std::vector<std::int32_t> mask;
    //! @brief The number of object classes.
    std::int32_t classes;

    std::int32_t num;
    float jitter;
    float scale_x_y;
    float cls_normalizer;
    float iou_normalizer;
    std::string iou_loss;
    float ignore_thresh;
    float truth_thresh;
    int random;
    float resize;
    std::string nms_kind;
    float beta_nms;

    inline auto
    update_output_sizes(const std::vector<std::unique_ptr<Layer>>& nodes)
    {
      output_sizes = (*(nodes.rbegin() + 1))->output_sizes;
      output_sizes[1] = 255;
      output.resize(output_sizes);
    }

    inline auto parse_line(const std::string& line) -> void override
    {
      auto line_split = std::vector<std::string>{};
      boost::split(line_split, line, boost::is_any_of("="),
                   boost::token_compress_on);
      for (auto& str : line_split)
        boost::trim(str);

      const auto& key = line_split[0];
      if (key == "mask")
      {
        auto mask_strings = std::vector<std::string>{};
        boost::split(mask_strings, line_split[1], boost::is_any_of(", "),
                     boost::token_compress_on);
        for (const auto& mask_str : mask_strings)
          mask.push_back(std::stoi(mask_str));
      }
      else if (key == "anchors")
      {
        auto anchors_strings = std::vector<std::string>{};
        boost::split(anchors_strings, line_split[1], boost::is_any_of(", "),
                     boost::token_compress_on);
        for (const auto& anchors_str : anchors_strings)
          anchors.push_back(std::stoi(anchors_str));
      }
      else if (key == "classes")
        classes = std::stoi(line_split[1]);
      else if (key == "num")
        num = std::stoi(line_split[1]);
      else if (key == "jitter")
        jitter = std::stof(line_split[1]);
      else if (key == "scale_x_y")
        scale_x_y = std::stof(line_split[1]);
      else if (key == "cls_normalizer")
        cls_normalizer = std::stof(line_split[1]);
      else if (key == "iou_normalizer")
        iou_normalizer = std::stof(line_split[1]);
      else if (key == "iou_loss")
        iou_loss = line_split[1];
      else if (key == "ignore_thresh")
        ignore_thresh = std::stof(line_split[1]);
      else if (key == "truth_thresh")
        truth_thresh = std::stof(line_split[1]);
      else if (key == "random")
        random = std::stoi(line_split[1]);
      else if (key == "resize")
        resize = std::stof(line_split[1]);
      else if (key == "nms_kind")
        nms_kind = line_split[1];
      else if (key == "beta_nms")
        beta_nms = std::stof(line_split[1]);
    }

    inline auto to_output_stream(std::ostream& os) const -> void override
    {
      os << "- mask           = ";
      std::copy(mask.begin(), mask.end(),
                std::ostream_iterator<int>(std::cout, ", "));
      os << "\n";

      os << "- anchors        = ";
      std::copy(anchors.begin(), anchors.end(),
                std::ostream_iterator<int>(std::cout, ", "));
      os << "\n";

      os << "- classes        = " << classes << "\n";
      os << "- num            = " << num << "\n";
      os << "- jitter         = " << jitter << "\n";
      os << "- scale_x_y      = " << scale_x_y << "\n";
      os << "- cls_normalizer = " << cls_normalizer << "\n";
      os << "- iou_normalizer = " << iou_normalizer << "\n";
      os << "- iou_loss       = " << iou_loss << "\n";
      os << "- ignore_thresh  = " << ignore_thresh << "\n";
      os << "- truth_thresh   = " << truth_thresh << "\n";
      os << "- random         = " << random << "\n";
      os << "- resize         = " << resize << "\n";
      os << "- nms_kind       = " << nms_kind << "\n";
      os << "- beta_nms       = " << beta_nms << "\n";

      os << "- input          = " << input_sizes.transpose() << "\n";
      os << "- output         = " << output_sizes.transpose() << "\n";
    }

    virtual auto forward(const TensorView_<float, 4>& x)
        -> const TensorView_<float, 4>& override
    {
      // Specific to COCO dataset.
      static constexpr auto num_classes = 80;     // Due to COCO
      static constexpr auto num_coordinates = 4;  // (x, y, w, h)
      static constexpr auto num_probabilities =
          1 /* P[object] */ + num_classes /* P[class|object] */;
      static constexpr auto num_box_features =
          num_coordinates + num_probabilities;

      static_assert(num_probabilities == 81);
      static_assert(num_box_features == 85);

      // Logistic activation function.
      const auto logistic_fn = [](float v) { return 1 / (1 + std::exp(-v)); };

      auto& y = output;

      const auto& alpha = scale_x_y;
      const auto beta = -0.5f * (alpha - 1);

      for (auto n = 0; n < x.size(0); ++n)
      {
        const auto xn = x[n];
        auto yn = y[n];

        for (auto box = 0; box < 3; ++box)
        {
          const auto x_channel = box * num_box_features + 0;
          const auto y_channel = box * num_box_features + 1;
          const auto w_channel = box * num_box_features + 2;
          const auto h_channel = box * num_box_features + 3;

          // Logistic activation in the (x, y) position.
          // That means (x, y) is in the cell.
          yn[x_channel].flat_array() =
              xn[x_channel].flat_array().unaryExpr(logistic_fn);
          yn[y_channel].flat_array() =
              xn[y_channel].flat_array().unaryExpr(logistic_fn);

          // Just copy.
          yn[w_channel].flat_array() = xn[w_channel].flat_array();
          yn[h_channel].flat_array() = xn[h_channel].flat_array();

          // Logistic activation for the probabilities.
          const auto p_start = box * num_box_features + num_coordinates;
          const auto p_end = (box + 1) * num_box_features;
          for (auto p = p_start; p < p_end; ++p)
            yn[p].flat_array() = xn[p].flat_array().unaryExpr(logistic_fn);
        }

        // Shift and rescale the position.
        for (auto box = 0; box < 3; ++box)
        {
          const auto x_channel = box * num_box_features + 0;
          const auto y_channel = box * num_box_features + 1;

          yn[x_channel].flat_array() =
              yn[x_channel].flat_array() * alpha + beta;
          yn[y_channel].flat_array() =
              yn[y_channel].flat_array() * alpha + beta;
        }
      }

      return output;
    }
  };

}  // namespace DO::Sara::Darknet
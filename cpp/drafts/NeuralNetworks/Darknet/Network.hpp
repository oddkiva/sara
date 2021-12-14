#pragma once

#include <DO/Sara/Core/MultiArray/Slice.hpp>
#include <DO/Sara/Core/TicToc.hpp>

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>

#include <iomanip>


namespace DO::Sara::Darknet {

  struct Network
  {
    using TensorView = TensorView_<float, 4>;

    inline auto forward_to_conv(Darknet::Convolution& conv, int i) -> void
    {
      const auto& x = net[i - 1]->output;
      auto& y = conv.output;

      const auto& w = conv.weights.w;
      const auto& b = conv.weights.b;
      const auto& stride = conv.stride;
      const auto offset = -conv.size / 2;

      if (profile)
        tic();

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

      if (conv.activation == "leaky")
        y.cwise_transform_inplace([](float& x) { x = x > 0 ? x : 0.1f * x; });
      else if (conv.activation == "linear")
      {
      }
      else
        throw std::runtime_error{"Unsupported activation!"};

      if (profile)
        toc("Conv");
    }

    inline auto forward_to_route(Darknet::Route& route, int i) -> void
    {
      auto& y = route.output;

      if (route.layers.size() == 1)
      {
        if (profile)
          tic();

        const auto& rel_idx = route.layers.front();
        const auto glob_idx =
            rel_idx < 0 ? i + rel_idx
                        : rel_idx + 1 /* because of the input layer. */;
        const auto& x = net[glob_idx]->output;

        auto start = Eigen::Vector4i{};
        if (route.group_id != -1)
          start << 0, route.group_id * (x.size(1) / route.groups), 0, 0;
        else
          start.setZero();

        const auto& end = x.sizes();


        const auto x_slice = slice(x, {
                                          {start(0), end(0), 1},
                                          {start(1), end(1), 1},
                                          {start(2), end(2), 1},
                                          {start(3), end(3), 1},
                                      });
        std::transform(std::begin(x_slice), std::end(x_slice), std::begin(y),
                       [](const auto& v) { return v; });

        if (profile)
          toc("Route-Group");
      }
      else
      {
        if (route.groups != 1)
        {
          std::cerr << "Error at route layer\n" << route << std::endl;
          std::cerr << "route groups = " << route.groups << std::endl;
          throw std::runtime_error{"Route layer implementation incomplete!"};
        }

        if (profile)
          tic();

        auto c_start = 0;
        auto c_end = 0;
        for (const auto& rel_idx : route.layers)
        {
          const auto glob_idx =
              rel_idx < 0 ? i + rel_idx
                          : rel_idx + 1 /* because of the input layer. */;
          const auto& x = net[glob_idx]->output;

          c_end += x.size(1);
          const auto y_start =
              (Eigen::Vector4i{} << 0, c_start, 0, 0).finished();
          const auto y_end =
              (Eigen::Vector4i{} << y.size(0), c_end, y.size(2), y.size(3))
                  .finished();

          for (auto n = 0; n < y.size(0); ++n)
            for (auto c = 0; c < x.size(1); ++c)
              y[n][c_start + c] = x[n][c];

          c_start = c_end;
        }

        if (profile)
          toc("Route-Concat");
      }
    }

    inline auto forward_to_maxpool(Darknet::MaxPool& maxpool, int i) -> void
    {
      const auto& x = net[i - 1]->output;
      auto& y = maxpool.output;
      if (maxpool.size != 2)
        throw std::runtime_error{
            "MaxPool implementation incomplete! size must be 2"};

      if (profile)
        tic();

      const auto& stride = maxpool.stride;
      const auto start = Eigen::Vector4i::Zero().eval();
      const auto& end = x.sizes();
      const auto steps = (Eigen::Vector4i{} << 1, 1, stride, stride).finished();

      const auto infx = make_infinite(x, make_constant_padding(0.f));
      auto xi = infx.begin_stepped_subarray(start, end, steps);
      auto yi = y.begin();

      for (; yi != y.end(); ++yi, ++xi)
      {
        const Matrix<int, 4, 1> s = xi.position();
        const Matrix<int, 4, 1> e =
            xi.position() + Eigen::Vector4i{1, 1, maxpool.size, maxpool.size};

        auto x_arr = std::array<float, 4>{};
        auto p = TensorView_<float, 4>{x_arr.data(), e - s};
        crop(p, infx, s, e);

        *yi = *std::max_element(p.begin(), p.end());
      }

      if (profile)
        toc("MaxPool");
    }

    inline auto forward_to_yolo(Darknet::Yolo& yolo, int i) -> void
    {
      if (profile)
        tic();

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

      const auto& x = net[i - 1]->output;
      auto& y = yolo.output;

      const auto& alpha = yolo.scale_x_y;
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

      if (profile)
        toc("YOLO forward pass");
    }

    inline auto forward_to_upsample(Darknet::Upsample& upsample, int i) -> void
    {
      const auto& x = net[i - 1]->output;
      auto& y = upsample.output;

      if (profile)
        tic();

      const auto& stride = upsample.stride;

      for (auto yi = y.begin_array(); yi != y.end(); ++yi)
      {
        Matrix<int, 4, 1> p = yi.position();
        p.tail(2) /= stride;
        *yi = x(p);
      }

      if (profile)
        toc("Upsample");
    }

    inline auto forward(const TensorView_<float, 4>& x) -> void
    {
      net[0]->output = x;
      for (auto i = 1u; i < net.size(); ++i)
      {
        if (debug)
          std::cout << "Forwarding to layer " << i << "\n"
                    << *net[i] << std::endl;

        if (auto conv = dynamic_cast<Convolution*>(net[i].get()))
          forward_to_conv(*conv, i);
        else if (auto route = dynamic_cast<Route*>(net[i].get()))
          forward_to_route(*route, i);
        else if (auto maxpool = dynamic_cast<MaxPool*>(net[i].get()))
          forward_to_maxpool(*maxpool, i);
        else if (auto upsample = dynamic_cast<Upsample*>(net[i].get()))
          forward_to_upsample(*upsample, i);
        else if (auto yolo = dynamic_cast<Yolo*>(net[i].get()))
          forward_to_yolo(*yolo, i);
        else
          break;

        if (debug)
          std::cout << std::endl;
      }
    }

    bool debug = false;
    bool profile = true;
    std::vector<std::unique_ptr<Layer>> net;
  };

}  // namespace DO::Sara::Darknet

#pragma once

#include <DO/Sara/Core/MultiArray/Slice.hpp>
#include <DO/Sara/Core/TicToc.hpp>

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>


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
      std::cout << "Forwarding to layer " << i << "\n" << conv << std::endl;

      tic();

      // Convolve.
      im2col_gemm_convolve(
          y,
          x,                                       // the signal
          w,                                       // the transposed kernel.
          make_constant_padding(0.f),              // the padding type
          {x.size(0), x.size(1), stride, stride},  // strides in the convolution
          {0, 0, -1, -1});  // Be careful about the C dimension.

      // Bias.
      for (auto n = 0; n < y.size(0); ++n)
        for (auto c = 0; c < y.size(1); ++c)
          y[n][c].flat_array() += b(c);

      if (conv.activation == "leaky")
        y.cwise_transform_inplace([](float& x) {
          if (x < 0)
            x *= 0.1f;
        });
      else if (conv.activation == "linear")
        std::cout << "linear activation" << std::endl;
      else
        throw std::runtime_error{"Unsupported activation!"};

      toc("Convolution Forward Pass");
      std::cout << std::endl;
    }

    inline auto forward_to_route(Darknet::Route& route, int i) -> void
    {
      std::cout << "Forwarding to layer " << i << "\n" << route << std::endl;

      auto& y = route.output;

      if (route.layers.size() == 1)
      {
        tic();

        const auto& rel_idx = route.layers.front();
        const auto glob_idx = rel_idx < 0 ? i + rel_idx : rel_idx;
        const auto& x = net[glob_idx]->output;

        auto start = Eigen::Vector4i{} ;
        if (route.group_id != -1)
          start << 0, route.group_id * (x.size(1) / route.groups), 0, 0;
        else
          start.setZero();

        const auto end = x.sizes();

#ifdef DEBUG_ROUTE
        std::cout << "start = " << start.transpose() << std::endl;
        std::cout << "end   = " << end.transpose() << std::endl;
#endif

        const auto x_slice = slice(x, {
                                          {start(0), end(0), 1},
                                          {start(1), end(1), 1},
                                          {start(2), end(2), 1},
                                          {start(3), end(3), 1},
                                      });
        std::transform(std::begin(x_slice), std::end(x_slice), std::begin(y),
                       [](const auto& v) { return v; });

        toc("Route Forward Pass");
        std::cout << std::endl;
      }
      else
      {
        if (route.groups != 1)
        {
          std::cerr << "Error at route layer\n" << route << std::endl;
          std::cout << "route groups = " << route.groups << std::endl;
          throw std::runtime_error{"Route layer implementation incomplete!"};
        }

        std::cout << "CONCATENATION" << std::endl;

        tic();

        auto c_start = 0;
        auto c_end = 0;
        for (const auto& rel_idx : route.layers)
        {
          const auto glob_idx = rel_idx < 0 ? i + rel_idx : rel_idx;
          std::cout << "Concatenating layer output: " << glob_idx << std::endl;
          const auto& x = net[glob_idx]->output;

          c_end += x.size(1);

          const auto y_start =
              (Eigen::Vector4i{} << 0, c_start, 0, 0).finished();
          const auto y_end =
              (Eigen::Vector4i{} << y.size(0), c_end, y.size(2), y.size(3))
                  .finished();

          std::cout << "x_start       = " << Eigen::Vector4i::Zero().transpose() << std::endl;
          std::cout << "x_end         = " << x.sizes().transpose() << std::endl;
          std::cout << "y_slice_start = " << y_start.transpose() << std::endl;
          std::cout << "y_slice_end   = " << y_end.transpose() << std::endl;
          std::cout << "y_start       = " << Eigen::Vector4i::Zero().transpose() << std::endl;
          std::cout << "y_end         = " << y.sizes().transpose() << std::endl;
          std::cout << std::endl;

          auto xi = x.begin();
          auto yi = y.begin_subarray(y_start, y_end);
          for ( ; !yi.end(); ++yi, ++xi)
            *yi = *xi;

          c_start = c_end;
        }
        toc("Route forward pass (CONCATENATION)");
        std::cout << std::endl;
      }
    }

    inline auto forward_to_maxpool(Darknet::MaxPool& maxpool, int i) -> void
    {
      std::cout << "Forwarding to layer " << i << "\n" << maxpool << std::endl;

      const auto& x = net[i - 1]->output;
      auto& y = maxpool.output;
      if (maxpool.size != 2)
        throw std::runtime_error{
            "MaxPool implementation incomplete! size must be 2"};

      tic();

      const Eigen::Vector4i start = Eigen::Vector4i::Zero();
      const Eigen::Vector4i end = x.sizes();
      const auto stride = maxpool.stride;
      const Eigen::Vector4i steps =
          (Eigen::Vector4i{} << 1, 1, stride, stride).finished();

      const auto infx = make_infinite(x, make_constant_padding(0.f));
      auto xi = infx.begin_stepped_subarray(start, end, steps);
      auto yi = y.begin();

      for ( ; yi != y.end(); ++yi, ++xi)
      {
        auto x_arr = std::array<float, 4>{};
        const Matrix<int, 4, 1> s = xi.position();
        const Matrix<int, 4, 1> e = xi.position() + Eigen::Vector4i{1, 1, maxpool.size, maxpool.size};

        auto p = TensorView_<float, 4>{x_arr.data(), e - s};
        crop(p, infx, s, e);

        *yi = *std::max_element(x_arr.begin(), x_arr.end());

#ifdef DEBUG_MAXPOOL
        std::cout << "p = " << xi.position().transpose() << std::endl;
        std::cout << "s = " << s.transpose() << "    ";
        std::cout << "e = " << e.transpose() << std::endl;
        std::cout << "p.sizes() = " << p.sizes().transpose() << std::endl;
        std::cout << "p.matrix() =\n" << p[0][0].matrix() << std::endl;
        std::cout << "*yi = " << *yi << std::endl;
        std::cout << std::endl;
#endif
      }

      toc("MaxPool Forward Pass");
      std::cout << std::endl;
    }

    inline auto forward(const TensorView_<float, 4>& x) -> void
    {
      net[0]->output = x;
      for (auto i = 1u; i < net.size(); ++i)
      {
        if (auto conv = dynamic_cast<Convolution*>(net[i].get()))
          forward_to_conv(*conv, i);
        else if (auto route = dynamic_cast<Route*>(net[i].get()))
          forward_to_route(*route, i);
        else if (auto maxpool = dynamic_cast<MaxPool*>(net[i].get()))
          forward_to_maxpool(*maxpool, i);
        else
          break;
      }
    }

    std::vector<std::unique_ptr<Layer>> net;
  };

}  // namespace DO::Sara::Darknet

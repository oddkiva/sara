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
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      conv.forward(x);

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
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      maxpool.forward(x);

      if (profile)
        toc("MaxPool");
    }

    inline auto forward_to_yolo(Darknet::Yolo& yolo, int i) -> void
    {
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      yolo.forward(x);

      if (profile)
        toc("YOLO forward pass");
    }

    inline auto forward_to_upsample(Darknet::Upsample& upsample, int i) -> void
    {
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      upsample.forward(x);

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

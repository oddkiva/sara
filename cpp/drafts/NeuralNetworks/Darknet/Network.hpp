// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2022 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#pragma once

#include <DO/Sara/Core/MultiArray/Slice.hpp>
#include <DO/Sara/Core/TicToc.hpp>

#include <DO/Sara/ImageProcessing/GemmBasedConvolution.hpp>

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>

#include <iomanip>
#include <optional>


namespace DO::Sara::Darknet {

  struct Network
  {
    using TensorView = TensorView_<float, 4>;

    auto get_input(int i) -> TensorView
    {
      if (i <= 0)
        throw std::runtime_error{"Input index must be positive!"};

      return net[i - 1]->output;
    }

    auto get_output(int i) -> TensorView
    {
      if (i < 0)
        throw std::runtime_error{"Input index must be positive!"};

      return net[i]->output;
    }

    auto forward_to_conv(Darknet::Convolution& conv, int i) -> void
    {
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      conv.forward(x);

      if (profile)
        toc("Conv");
    }

    auto forward_to_route(Darknet::Route& route, int i) -> void
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
          toc("Route-Slice");
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
          for (auto n = 0; n < y.size(0); ++n)
            for (auto c = 0; c < x.size(1); ++c)
              y[n][c_start + c] = x[n][c];

          c_start = c_end;
        }

        if (profile)
          toc("Route-Concat");
      }
    }

    auto forward_to_maxpool(Darknet::MaxPool& maxpool, int i) -> void
    {
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      maxpool.forward(x);

      if (profile)
        toc("MaxPool");
    }

    auto forward_to_yolo(Darknet::Yolo& yolo, int i) -> void
    {
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      yolo.forward(x);

      if (profile)
        toc("YOLO forward pass");
    }

    auto forward_to_upsample(Darknet::Upsample& upsample, int i) -> void
    {
      if (profile)
        tic();

      const auto& x = net[i - 1]->output;
      upsample.forward(x);

      if (profile)
        toc("Upsample");
    }

    auto forward_to_shortcut(Darknet::Shortcut& shortcut, int i) -> void
    {
      if (profile)
        tic();

      const auto i1 = i - 1;
      const auto i2 = shortcut.from < 0  //
                          ? i + shortcut.from
                          : shortcut.from;
      const auto& fx = net[i1]->output;
      const auto& x = net[i2]->output;
      shortcut.forward(fx, x);

      if (profile)
        toc("Shortcut");
    }

    auto forward(const TensorView_<float, 4>& x,
                 std::optional<std::size_t> up_to_layer_idx = std::nullopt)
        -> void
    {
      const auto n = up_to_layer_idx.has_value()  //
                         ? (*up_to_layer_idx + 1)
                         : net.size();

      net[0]->output = x;
      for (auto i = 1u; i < n; ++i)
      {
        if (debug)
          std::cout << "Forwarding to layer " << i << " (" << net[i]->type
                    << ")\n"
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
        else if (auto shortcut = dynamic_cast<Shortcut*>(net[i].get()))
          forward_to_shortcut(*shortcut, i);
        else
          throw std::runtime_error{"Layer[" + std::to_string(i) + "] = " +
                                   net[i]->type + " is not implemented!"};

        if (debug)
          std::cout << std::endl;
      }
    }

    bool debug = false;
    bool profile = true;
    std::vector<std::unique_ptr<Layer>> net;
  };

}  // namespace DO::Sara::Darknet

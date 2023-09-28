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

#include <drafts/NeuralNetworks/Darknet/Network.hpp>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>

#include <boost/filesystem.hpp>

#include <string>


namespace DO::Sara::Darknet {

  // CAVEAT: this is sensitive to the CPU architecture endianness.
  inline auto read_tensor(const std::string& filepath) -> Tensor_<float, 4>
  {
    auto file = std::ifstream{filepath, std::ios::binary};
    if (!file.is_open())
      throw std::runtime_error{"Error: could not open file: " + filepath + "!"};

    auto sizes = Eigen::Vector4i{};
    file.read(reinterpret_cast<char*>(sizes.data()), sizeof(float) * 4);

    auto output = Tensor_<float, 4>{sizes};
    const auto num_elements = std::accumulate(sizes.data(), sizes.data() + 4, 1,
                                              std::multiplies<int>());
    file.read(reinterpret_cast<char*>(output.data()),
              sizeof(float) * num_elements);

    return output;
  }

  // CAVEAT: this is sensitive to the CPU architecture endianness.
  inline auto read_all_intermediate_outputs(const std::string& dir_path)
  {
    namespace fs = boost::filesystem;

    auto stringify = [](int n) {
      std::ostringstream ss;
      ss << std::setw(3) << std::setfill('0') << n;
      return ss.str();
    };

    auto outputs = std::vector<Tensor_<float, 4>>(38);
    for (auto i = 0u; i < outputs.size(); ++i)
    {
      const auto filepath = fs::path{dir_path} / (stringify(i) + ".bin");
      std::cout << "Parsing " << filepath << std::endl;
      outputs[i] = Darknet::read_tensor(filepath.string());
    }

    return outputs;
  }


  inline auto visualize_layer_output(const TensorView_<float, 4>& y,  //
                                     const Eigen::Vector2i& sizes)
  {
    for (auto i = 0; i < y.size(1); ++i)
    {
      const auto y_i = y[0][i];
      const auto im_i = image_view(y_i);
      const auto im_i_rescaled = resize(color_rescale(im_i), sizes);
      display(im_i_rescaled);
      get_key();
    }
  }


  inline auto check_against_ground_truth(
      const TensorView_<float, 4>& gt,  // ground-truth
      const TensorView_<float, 4>& me,  // my implementation
      const Eigen::Vector2i& sizes)
  {
    auto reformat = [&sizes](const auto& y) {
      const auto y_i = y;
      const auto im_i = image_view(y_i);
      const auto im_i_rescaled = resize(color_rescale(im_i), sizes);
      return im_i_rescaled;
    };

    for (auto i = 0; i < gt.size(1); ++i)
    {
      // Calculate on the actual tensor.
      auto diff = Tensor_<float, 2>{gt[0][i].sizes()};
      diff.matrix() = gt[0][i].matrix() - me[0][i].matrix();

      const auto residual = diff.matrix().norm();
      const auto min_diff = diff.matrix().cwiseAbs().minCoeff();
      const auto max_diff = diff.matrix().cwiseAbs().maxCoeff();

      if (false)
      {
        std::cout << "residual " << i << " = " << residual << std::endl;
        std::cout << "min residual value " << i << " = " << min_diff
                  << std::endl;
        std::cout << "max residual value " << i << " = " << max_diff
                  << std::endl;

        std::cout << "GT\n" << gt[0][i].matrix().block(0, 0, 5, 5) << std::endl;
        std::cout << "ME\n" << me[0][i].matrix().block(0, 0, 5, 5) << std::endl;
      }

      if (max_diff > 6e-5f)
      {
        // Resize and color rescale the data to show it nicely.
        const auto im1 = reformat(gt[0][i]);
        const auto im2 = reformat(me[0][i]);
        const auto imdiff = reformat(diff);

        display(im1);
        display(im2, {im1.width(), 0});
        display(imdiff, {2 * im1.width(), 0});

        get_key();
        throw std::runtime_error{"FISHY COMPUTATION ERROR!"};
      }
    }
  }

  inline auto check_convolutional_weights(const Network& model,
                                          const std::string& data_dirpath)
      -> void
  {
    const auto stringify = [](int n) {
      std::ostringstream ss;
      ss << std::setw(3) << std::setfill('0') << n;
      return ss.str();
    };

    const auto& net = model.net;
    for (auto i = 0u; i < net.size(); ++i)
    {
      if (auto conv = dynamic_cast<const Convolution*>(net[i].get()))
      {
        const auto weights_fp =
            data_dirpath + "/kernel-" + stringify(i - 1) + ".bin";
        const auto biases_fp =
            data_dirpath + "/bias-" + stringify(i - 1) + ".bin";

        const auto w = read_tensor(weights_fp).reshape(conv->weights.w.sizes());
        const auto b = read_tensor(biases_fp);

        const auto diffb = (conv->weights.b - b.vector()).norm();
        const auto diffw = (conv->weights.w.vector() - w.vector()).norm();

        if (diffb > 5e-6f || diffw > 5e-6f)
        {
          std::cout << i << " diffb = " << diffb << std::endl;
          std::cout << i << " diffw = " << diffw << std::endl;
          throw std::runtime_error{"Error: convolutional weights are wrong!"};
        }
      }
    }
  }

  inline auto check_yolov4_tiny_implementation(Network& model,
                                               const std::string& output_dir)
  {
    namespace fs = boost::filesystem;

    if (!fs::exists(output_dir))
      throw std::runtime_error{"Ouput directory " + output_dir +
                               "does not exist!"};

    // Check the weights.
    check_convolutional_weights(model, output_dir);

    const auto x = Darknet::read_tensor(               //
        (fs::path{output_dir} / "input.bin").string()  //
    );
    const auto xt = x.transpose({0, 2, 3, 1});

    const auto image = ImageView<Rgb32f>{
        reinterpret_cast<Rgb32f*>(const_cast<float*>(xt.data())),
        {xt.size(2), xt.size(1)}};
    const auto& image_resized = image;

    create_window(3 * image.width(), image.height());
    display(image);
    get_key();

    model.debug = true;
    model.forward(x);

    // Compare my layer outputs with Darknet's.
    const auto gt = read_all_intermediate_outputs(output_dir);

    const auto& net = model.net;
    for (auto layer = 1u; layer < net.size(); ++layer)
    {
      std::cout << "CHECKING LAYER " << layer << ": " << net[layer]->type
                << std::endl
                << *net[layer] << std::endl;
      check_against_ground_truth(gt[layer - 1], net[layer]->output,
                                 image_resized.sizes());
    }
  }

}  // namespace DO::Sara::Darknet

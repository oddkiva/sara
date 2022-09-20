#pragma once

#include <drafts/NeuralNetworks/TensorRT/Helpers.hpp>


namespace DO::Sara::TensorRT {

  class Network
  {
  public:
    using model_weights_dict_type = std::map<std::string, std::vector<float>>;

    explicit Network() = default;

    auto model()
    {
      return _model;
    }

    auto weights() -> model_weights_dict_type&
    {
      return _model_weights;
    }

    static auto set_current(Network& net)
    {
      _current_network = &net;
    }

    static auto current() -> Network&
    {
      if (_current_network == nullptr)
        throw std::runtime_error{"Error: the current network is invalid!"};

      return *_current_network;
    }

    static auto current_var_index() -> int
    {
      return _current_var_index;
    }

    static auto increment_current_var_index() -> int
    {
      return ++_current_var_index;
    }

  private:
    static Network* _current_network;
    static int _current_var_index;

  private:
    nvinfer1::INetworkDefinition* _model = nullptr;
    model_weights_dict_type _model_weights;
  };

  Network* Network::_current_network = nullptr;
  int Network::_current_var_index = 0;


  inline auto make_placeholder(const std::array<int, 3>& chw_sizes,
                               const std::string& name = "image")
      -> const nvinfer1::ITensor&
  {
    auto net = Network::current().model();

    const auto data = net->addInput(
        name.c_str(), nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims3{chw_sizes[0], chw_sizes[1], chw_sizes[2]});

    return *data;
  }

  inline auto make_placeholder(const std::array<int, 4>& nchw_sizes,
                               const std::string& name = "image")
      -> nvinfer1::ITensor&
  {
    auto net = Network::current().model();

    const auto data =
        net->addInput(name.c_str(), nvinfer1::DataType::kFLOAT,
                      nvinfer1::Dims4{nchw_sizes[0], nchw_sizes[1],
                                      nchw_sizes[2], nchw_sizes[3]});

    return *data;
  }


  inline auto operator*(nvinfer1::ITensor& x,
                        const std::pair<float, std::string>& y)
      -> nvinfer1::ITensor&
  {
    auto model = Network::current().model();
    auto weights = Network::current().weights();

    const auto [y_value, y_name] = y;
    weights[y_name] = {y_value};

    const auto y_weights =
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                          reinterpret_cast<const void*>(weights[y_name].data()),
                          static_cast<std::int64_t>(weights[y_name].size())};
    const auto scale_op = model->addScale(x, nvinfer1::ScaleMode::kELEMENTWISE,
                                          {}, y_weights, {});

    auto x_div_y = scale_op->getOutput(0);

    return *x_div_y;
  }

  inline auto operator/(nvinfer1::ITensor& x,
                        const std::pair<float, std::string>& y)
      -> nvinfer1::ITensor&
  {
    return x * make_pair(1 / y.first, y.second);
  }

  inline auto operator*(nvinfer1::ITensor& x, float y) -> nvinfer1::ITensor&
  {
    const auto y_name =
        "weights/" + std::to_string(Network::current_var_index());
    Network::increment_current_var_index();
    return x * std::make_pair(y, y_name);
  }

  inline auto operator/(nvinfer1::ITensor& x, float y) -> nvinfer1::ITensor&
  {
    return x * (1 / y);
  }

  inline auto conv_2d(nvinfer1::ITensor& x,              //
                      int num_filters,                   //
                      nvinfer1::DimsHW kernel_sizes,     //
                      const std::vector<float>& w = {},  //
                      const std::vector<float>& b = {},
                      const std::string& name = {})  //
      -> nvinfer1::ITensor&
  {
    auto model = Network::current().model();
    auto weights = Network::current().weights();

    auto w_name = std::string{};
    auto b_name = std::string{};
    if (name.empty())
    {
      w_name = "weights/" + std::to_string(Network::current_var_index());
      weights[w_name] = w;
      Network::increment_current_var_index();

      b_name = "weights/" + std::to_string(Network::current_var_index());
      weights[b_name] = b;
      Network::increment_current_var_index();
    }
    else
    {
      w_name = "conv/weights/" + std::to_string(Network::current_var_index());
      if (weights.find(w_name) != weights.end())
        throw std::runtime_error{
            "Error: convolution weight name already used!"};
      weights[w_name] = w;

      b_name = "conv/bias/" + std::to_string(Network::current_var_index());
      if (weights.find(b_name) != weights.end())
        throw std::runtime_error{"Error: convolution bias name already used!"};
      weights[b_name] = w;
    }

    const auto w_weights =
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                          reinterpret_cast<const void*>(weights[w_name].data()),
                          static_cast<std::int64_t>(weights[w_name].size())};
    const auto b_weights =
        nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                          reinterpret_cast<const void*>(weights[b_name].data()),
                          static_cast<std::int64_t>(weights[b_name].size())};

    const auto conv_op = model->addConvolutionNd(x, num_filters, kernel_sizes,
                                                 w_weights, b_weights);

    auto y = conv_op->getOutput(0);

    return *y;
  }

}  // namespace DO::Sara::TensorRT
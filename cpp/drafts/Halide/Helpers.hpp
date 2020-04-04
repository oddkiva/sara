#include "MyHalide.hpp"


namespace DO::Shakti::HalideBackend {

  using namespace Halide;

  template <typename T>
  inline auto identity(const Buffer<T>& in, Var& x, Var& y)
  {
    auto f = Func{"identity"};
    f(x, y) = BoundaryConditions::repeat_edge(in)(x, y);
    return f;
  }

  template <typename T, typename... Vars>
  inline auto identity(const Buffer<T>& in, Vars&... vars)
  {
    auto f = Func{"identity"};
    f(vars...) = in(vars...);
    return f;
  }

  template <typename T>
  inline auto shift(const Buffer<T>& in, Var& i, int t)
  {
    auto f = Func{"shift"};
    f(i) = in(i - t);
    return f;
  }

  inline auto transpose(Func& f, Var& x, Var& y)
  {
    auto ft = Func{f.name() + "_" + "transposed"};
    ft(x, y) = f(y, x);
    return ft;
  }

  inline auto conv_x(Func& signal, Func& filter, Var& x, Var& y,
                     RDom& summation_domain)
  {
    auto& k = summation_domain;

    auto g = Func{"conv_x_" + signal.name()};
    g(x, y) = sum(signal(x + k, y) * filter(k));
    return g;
  }

  inline auto conv_y(Func& signal, Func& filter, Var& x, Var& y,
                     RDom& summation_domain)
  {
    auto& k = summation_domain;
    auto g = Func{"conv_y_" + signal.name()};
    g(x, y) = sum(signal(x, y + k) * filter(k));
    return g;
  }

  inline auto separable_conv_2d(Func& signal, Func& kernel, Var& x, Var& y, RDom& k)
  {
    auto g = Func{};
    g = transpose(signal, x, y);
    g = conv_x(g, kernel, x, y, k);
    g = transpose(signal, x, y);
    g = conv_x(g, kernel, x, y, k);
    return g;
  }

  inline auto gaussian_unnormalized(Expr x) -> Func
  {
    auto g = Func{"gaussian_unnormalized"};
    g(x) = exp(-(x * x));
    return g;
  }

  inline auto gaussian(Var& x, Expr& sigma, RDom& summation_domain) -> Func
  {
    auto& k = summation_domain;
    auto gu = gaussian_unnormalized(x / sigma);
    auto normalization_factor = sum(gu(k));

    auto g = Func{"gaussian"};
    g(x) = gu(x) / normalization_factor;

    return g;
  }

  inline auto gaussian_conv_2d(Func& signal, Expr& sigma, Var& x, Var& y, Expr truncation_factor)
  {
    const auto radius = cast<int>(sigma / 2) * truncation_factor;
    auto k = RDom(-radius, 2 * radius + 1);
    auto g = gaussian(x, sigma, k);
    return separable_conv_2d(signal, g, x, y, k);
  }

}  // namespace DO::Shakti::HalideBackend

#pragma once

#include <DO/Sara/Core.hpp>


namespace DO::Sara {

  struct TicToc : public Timer
  {
    static auto instance() -> TicToc&
    {
      static TicToc _instance;
      return _instance;
    }
  };

  inline auto tic()
  {
    TicToc::instance().restart();
  }

  inline auto toc(const std::string& what)
  {
    const auto elapsed = TicToc::instance().elapsed_ms();
    std::cout << "[" << what << "] " << elapsed << " ms" << std::endl;
  }

}  // namespace DO::Sara

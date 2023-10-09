#pragma once

#include <DO/Sara/Core/EigenFormatInterop.hpp>

#include <boost/log/trivial.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>

#include <fmt/core.h>

#include <optional>


namespace DO::Sara {

  class Logger
  {
  private:
    Logger() = default;

  public:
    using Impl = boost::log::sources::severity_logger<
        boost::log::trivial::severity_level>;

    enum class SeverityLevel : std::uint8_t
    {
      Trace = 0,
      Debug = 1,
      Info = 2,
      Warning = 3,
      Error = 4
    };

    Logger(const Logger&) = delete;
    auto operator=(const Logger&) -> Logger& = delete;

    static auto
    init(const SeverityLevel level = SeverityLevel::Debug,
         const std::optional<std::string>& log_filepath = std::nullopt) -> void;

    static auto get() -> Impl&;

  private:
    Impl _impl;
    static bool _is_initialized;
  };

}  // namespace DO::Sara

#define SARA_LOG(log_, sv)                                                     \
  BOOST_LOG_SEV(log_, boost::log::trivial::sv)                                 \
      << boost::log::add_value("Line", __LINE__)                               \
      << boost::log::add_value("File", __FILE__)                               \
      << boost::log::add_value("Function", __FUNCTION__)


#define SARA_LOGT(log_, ...) SARA_LOG(log_, trace) << fmt::format(__VA_ARGS__)
#define SARA_LOGD(log_, ...) SARA_LOG(log_, debug) << fmt::format(__VA_ARGS__)
#define SARA_LOGI(log_, ...) SARA_LOG(log_, info) << fmt::format(__VA_ARGS__)
#define SARA_LOGW(log_, ...) SARA_LOG(log_, warning) << fmt::format(__VA_ARGS__)
#define SARA_LOGE(log_, ...) SARA_LOG(log_, error) << fmt::format(__VA_ARGS__)

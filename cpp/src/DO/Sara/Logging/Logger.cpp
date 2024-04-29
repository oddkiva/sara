#include <DO/Sara/Logging/Logger.hpp>

#include <boost/core/null_deleter.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>


namespace logging = boost::log;
namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;


namespace DO::Sara {

  static auto log_formatter(const boost::log::record_view& rec,
                            boost::log::formatting_ostream& strm) -> void
  {
#if !defined(__APPLE__)
    auto severity = rec[logging::trivial::severity];
    if (severity)
    {
      // Set the color
      switch (severity.get())
      {
      case logging::trivial::severity_level::debug:
      case logging::trivial::severity_level::info:
        strm << "\033[32m";
        break;
      case logging::trivial::severity_level::warning:
        strm << "\033[33m";
        break;
      case logging::trivial::severity_level::error:
      case logging::trivial::severity_level::fatal:
        strm << "\033[31m";
        break;
      default:
        break;
      }
    }
#endif

    // Get the LineID attribute value and put it into the stream
    strm << logging::extract<unsigned int>("LineID", rec) << ": ";
    const auto fullpath = logging::extract<std::string>("File", rec);
    strm << boost::filesystem::path(fullpath.get()).filename().string() << ": ";
    strm << logging::extract<std::string>("Function", rec) << ": ";
    strm << logging::extract<int>("Line", rec) << ": ";

    // The same for the severity level.
    // The simplified syntax is possible if attribute keywords are used.
    strm << "<" << rec[logging::trivial::severity] << "> ";

#if !defined(__APPLE__)
    // Restore the default color
    if (severity)
      strm << "\033[0m";
#endif

    // Finally, put the record message to the stream
    strm << rec[expr::smessage];
  }

  bool Logger::_is_initialized = false;

  auto Logger::init(const Logger::SeverityLevel level,
                    const std::optional<std::string>& log_filepath) -> void
  {
    if (_is_initialized)
      return;

    using TextSink = sinks::synchronous_sink<sinks::text_ostream_backend>;
    auto sink = boost::make_shared<TextSink>();

    // Store the log in the file if specified.
    if (log_filepath.has_value())
      sink->locked_backend()->add_stream(
          boost::make_shared<std::ofstream>(*log_filepath));

    // Otherwise log on the console by default.
    auto cout_stream =
        boost::shared_ptr<std::ostream>(&std::clog, boost::null_deleter());
    sink->locked_backend()->add_stream(cout_stream);

    sink->set_formatter(&log_formatter);

    logging::core::get()->add_sink(sink);

    boost::log::add_common_attributes();

    switch (level)
    {
    case SeverityLevel::Trace:
      logging::core::get()->set_filter(logging::trivial::severity >=
                                       logging::trivial::trace);
      break;
    case SeverityLevel::Debug:
      logging::core::get()->set_filter(logging::trivial::severity >=
                                       logging::trivial::debug);
      break;
    case SeverityLevel::Info:
      logging::core::get()->set_filter(logging::trivial::severity >=
                                       logging::trivial::info);
      break;
    case SeverityLevel::Warning:
      logging::core::get()->set_filter(logging::trivial::severity >=
                                       logging::trivial::warning);
      break;
    case SeverityLevel::Error:
      logging::core::get()->set_filter(logging::trivial::severity >=
                                       logging::trivial::error);
      break;
    default:
      break;
    }

    _is_initialized = true;
  }

  auto Logger::get() -> Logger::Impl&
  {
    // We didn't specify anything? Fair enough, output the log only on the
    // console.
    init();

    static auto instance = Logger{};
    return instance._impl;
  }

}  // namespace DO::Sara

#include <DO/Sara/SfM/ConfigParser.hpp>

#include <boost/property_tree/ini_parser.hpp>


namespace DO::Sara {

auto read_config(const std::string& ini_filepath) -> boost::property_tree::ptree
{
  auto pt = boost::property_tree::ptree{};
  boost::property_tree::ini_parser::read_ini(ini_filepath, pt);
  return pt;
}

} /* namespace DO::Sara */

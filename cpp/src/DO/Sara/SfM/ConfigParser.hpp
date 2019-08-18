#pragma once

#include <boost/property_tree/ptree.hpp>


namespace DO::Sara {

auto read_config(const std::string& ini_filepath)
    -> boost::property_tree::ptree;

} /* namespace DO::Sara */

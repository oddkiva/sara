#pragma once

#include <DO/Sara/Defines.hpp>

#include <boost/property_tree/ptree.hpp>


namespace DO::Sara {

DO_SARA_EXPORT
auto read_config(const std::string& ini_filepath)
    -> boost::property_tree::ptree;

} /* namespace DO::Sara */

// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#define BOOST_TEST_MODULE "Core/Serialization"

#include <DO/Sara/Core/Serialization.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <fstream>


using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_tensor_serialization)
{
  auto t_write = Tensor_<float, 3>{{2, 3, 4}};
  for (auto i = 0u; i < t_write.size(); ++i)
    t_write.data()[i] = static_cast<float>(i);

  const auto file =
      (boost::filesystem::temp_directory_path() / "tensor").string();

#ifdef DEBUG
  static_assert(std::is_same<decltype(file), const std::string>::value, "");
  std::cout << "temporary file = " << file << std::endl;
#endif

  std::ofstream ofs{file};
  {
    boost::archive::binary_oarchive oa{ofs};
    oa << t_write;
  }

  auto t_read = Tensor_<float, 3>{{2, 5, 1}};
  std::ifstream ifs{file};
  {
    boost::archive::binary_iarchive ia{ifs};
    ia >> t_read;
  }

  BOOST_CHECK(t_read.vector() == t_write.vector());
}

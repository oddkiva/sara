// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#include <DO/Sara/Core/H5.hpp>


using namespace DO::Sara;


void write_sfm_data()
{
  const auto filename = "sfm_dummy_data.h5";
  auto h5file = H5File{filename, H5F_ACC_TRUNC};

  const auto group = h5file.group("0");

  // Create dummy features.
  auto features = Tensor_<OERegion, 1>{4};
  for (int i = 0; i < 4; ++i)
  {
    features.data()[i].center() << i, i;
    features.data()[i].shape_matrix = DO::Sara::Matrix2f::Ones() * (i + 0.5f);
    features.data()[i].orientation = 30 * i;
    features.data()[i].extremum_value = 10 * i;
  }

  auto dataset = h5file.write_dataset("0/features", features);
  std::cout << "WRITE OK" << std::endl;
}


void read_sfm_data()
{
  const auto filename = "sfm_dummy_data.h5";
  auto h5file = H5File{filename, H5F_ACC_RDONLY};

  auto features = Tensor_<OERegion, 1>{};
  h5file.read_dataset("0/features", features);

  auto farray = features.flat_array();

  for (int i = 0; i < farray.size(); ++i)
  {
    std::cout << "index " << i << std::endl;
    std::cout << "center = \n" << farray(i).center() << std::endl;
    std::cout << "shape_matrix = \n" << farray(i).shape_matrix << std::endl;
    std::cout << "orientation = " << farray(i).orientation << std::endl;
    std::cout << "extremum value = " << farray(i).extremum_value << std::endl;
    std::cout << std::endl;
  }
}


int main()
{
  write_sfm_data();
  read_sfm_data();
  return 0;
}

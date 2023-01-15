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

#define BOOST_TEST_MODULE "MultiViewGeometry/HDF5 I/O"

#include <DO/Sara/Core/Math/Rotation.hpp>
#include <DO/Sara/MultiViewGeometry/HDF5.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
using namespace DO::Sara;


BOOST_AUTO_TEST_CASE(test_hdf5_read_write_pinhole_camera)
{
  const auto filepath = (fs::temp_directory_path() / "test.h5").string();

  // Write data to HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_TRUNC};

    auto cameras = std::vector<BasicPinholeCamera>{
        // K, R, t.
        {Matrix3d::Identity(), roll(0.), Vector3d::Zero()},
        {Matrix3d::Identity() * 2., pitch(1.), Vector3d::Ones()},
    };
    h5file.write_dataset("cameras", tensor_view(cameras));
  }

  // Read data from from HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};
    auto cameras = std::vector<BasicPinholeCamera>{};
    h5file.read_dataset("cameras", cameras);

    const auto& C0 = cameras[0];
    const auto& C1 = cameras[1];

    BOOST_CHECK_EQUAL(C0.K, Matrix3d::Identity());
    BOOST_CHECK_EQUAL(C0.R, roll(0.));
    BOOST_CHECK_EQUAL(C0.t, Vector3d::Zero());

    BOOST_CHECK_EQUAL(C1.K, Matrix3d::Identity() * 2.);
    BOOST_CHECK_EQUAL(C1.R, pitch(1.));
    BOOST_CHECK_EQUAL(C1.t, Vector3d::Ones());
  }

  fs::remove(filepath);
}

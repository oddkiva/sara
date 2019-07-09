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

#include <DO/Sara/MultiViewGeometry/HDF5.hpp>
#include <DO/Sara/MultiViewGeometry/Utilities.hpp>

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

    auto cameras = std::vector<PinholeCamera>{
        // K, R, t.
        {Matrix3d::Identity(), rotation_x(0.), Vector3d::Zero()},
        {Matrix3d::Identity() * 2., rotation_y(1.), Vector3d::Ones()},
    };
    h5file.write_dataset("cameras", tensor_view(cameras));
  }

  // Read data from from HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};
    auto cameras = std::vector<PinholeCamera>{};
    h5file.read_dataset("cameras", cameras);

    const auto& C0 = cameras[0];
    const auto& C1 = cameras[1];

    BOOST_CHECK_EQUAL(C0.K, Matrix3d::Identity());
    BOOST_CHECK_EQUAL(C0.R, rotation_x(0.));
    BOOST_CHECK_EQUAL(C0.t, Vector3d::Zero());

    BOOST_CHECK_EQUAL(C1.K, Matrix3d::Identity() * 2.);
    BOOST_CHECK_EQUAL(C1.R, rotation_y(1.));
    BOOST_CHECK_EQUAL(C1.t, Vector3d::Ones());
  }

  fs::remove(filepath);
}

BOOST_AUTO_TEST_CASE(test_hdf5_read_write_epipolar_edge)
{
  const auto filepath = (fs::temp_directory_path() / "test.h5").string();

  // Write data to HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_TRUNC};

    auto f_edges = std::vector<EpipolarEdge>{
        // i j F
        {0, 1, Matrix3d::Zero()},
        {1, 2, Matrix3d::Ones()},
    };
    h5file.write_dataset("cameras", tensor_view(f_edges));
  }

  // Read data from from HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};
    auto f_edges = std::vector<EpipolarEdge>{};
    h5file.read_dataset("cameras", f_edges);

    const auto& e0 = f_edges[0];
    const auto& e1 = f_edges[1];

    BOOST_CHECK_EQUAL(e0.i, 0);
    BOOST_CHECK_EQUAL(e0.j, 1);
    BOOST_CHECK_EQUAL(e0.m, Matrix3d::Zero());

    BOOST_CHECK_EQUAL(e1.i, 1);
    BOOST_CHECK_EQUAL(e1.j, 2);
    BOOST_CHECK_EQUAL(e1.m, Matrix3d::Ones());
  }

  fs::remove(filepath);
}

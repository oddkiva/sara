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

#define BOOST_TEST_MODULE "Core/HDF5 I/O"

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;
using namespace DO::Sara;

BOOST_AUTO_TEST_CASE(test_hdf5_read_write_array)
{
  const auto filepath = (fs::temp_directory_path() / "test.h5").string();

  // Write data to HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_TRUNC};

    h5file.group("some_group");

    auto array = Tensor_<float, 2>{3, 2};
    array.matrix() <<
      1, 2,
      3, 4,
      5, 6;
    h5file.write_dataset("some_group/array", array);
  }

  // Read data from from HDF5.
  {
    auto h5file = H5File{filepath, H5F_ACC_RDONLY};

    auto array = Tensor_<float, 2>{3, 2};
    h5file.read_dataset("some_group/array", array);

    auto true_matrix = MatrixXf(3, 2);
    true_matrix <<
      1, 2,
      3, 4,
      5, 6;

    BOOST_CHECK_EQUAL(array.matrix(), true_matrix);
  }

  fs::remove(filepath);
}


//BOOST_AUTO_TEST_CASE(test_hdf5_read_write_std_string)
//{
//  const auto filepath = (fs::temp_directory_path() / "test.h5").string();
//
//  constexpr auto SPACE1_DIM1 = 4;
//  constexpr auto SPACE1_RANK = 1;
//  const char* DSET_VLSTR_NAME = "vlstr_type";
//
//  // Write data to HDF5.
//  {
//    auto h5file = H5File{filepath, H5F_ACC_TRUNC};
//
//    const char* wdata[SPACE1_DIM1] = {
//        "Four score and seven years ago our forefathers brought forth on this "
//        "continent a new nation,",
//        "conceived in liberty and dedicated to the proposition that all men "
//        "are created equal.",
//        "Now we are engaged in a great civil war,",
//        "testing whether that nation or any nation so conceived and so "
//        "dedicated can long endure."}; /* Information to write */
//
//    /* Create dataspace for datasets */
//    hsize_t dims1[] = {SPACE1_DIM1};
//    H5::DataSpace sid1(SPACE1_RANK, dims1);
//
//    /* Create a datatype to refer to */
//    H5::StrType tid1(0, H5T_VARIABLE);
//
//    if (H5T_STRING != H5Tget_class(tid1.getId()) ||
//        !H5Tis_variable_str(tid1.getId()))
//      std::cerr << "this is not a variable length string type!!!" << std::endl;
//
//    /* Create a dataset */
//    H5::DataSet dataset = h5file.file->createDataSet("text_array", tid1, sid1);
//
//    /* Write dataset to disk */
//    dataset.write((void*) wdata, tid1);
//
//    /* Close Dataset */
//    dataset.close();
//  }
//
//  // Read data from from HDF5.
//  {
//    auto h5file = H5File{filepath, H5F_ACC_RDONLY};
//
//    char* rdata[SPACE1_DIM1]; /* Information read in */
//    hid_t native_type;        /* Datatype ID */
//  }
//
//  fs::remove(filepath);
//}

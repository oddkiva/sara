#define BOOST_TEST_MODULE "HDF5 I/O"

#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/Core/Tensor.hpp>


#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>


namespace fs = boost::filesystem;
using namespace DO::Sara;

BOOST_AUTO_TEST_CASE(test_hdf5_read_write_array)
{
  using tuple_type = std::tuple<int, std::string>;

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

    std::cout << array.matrix() << std::endl;
  }

  fs::remove(filepath);
}

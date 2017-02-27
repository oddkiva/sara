#include <DO/Sara/Core.hpp>

#include <hdf5/serial/H5Cpp.h>


int main()
{
  const auto filepath = "matrix.h5";
  const auto data_set_name = "single_matrix";
  const auto M = 5;
  const auto N = 6;

  auto file = H5::H5File{filepath, H5F_ACC_TRUNC};
  hsize_t dims[] = {M, N};

  const auto data_space = H5::DataSpace{2, dims};
  const auto data_set = file.createDataSet(data_set_name, H5::PredType::NATIVE_DOUBLE,
                                          data_space);


  auto m = DO::Sara::MatrixXd{M, N};
  std::iota(m.data(), m.data() + m.size(), 0.);

  data_set.write(m.data(), H5::PredType::NATIVE_DOUBLE);

  return 0;
}

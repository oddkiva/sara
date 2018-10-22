#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#include <hdf5/serial/H5Cpp.h>

#include <array>


constexpr auto filepath = "matrix.h5";
constexpr auto dataset_name = "single_matrix";
constexpr auto rank = 2;
constexpr auto M = 5;
constexpr auto N = 6;


void write_data()
{
  auto file = H5::H5File{filepath, H5F_ACC_TRUNC};
  auto dims = std::array<hsize_t, rank>{M, N};

  const auto dataspace = H5::DataSpace{rank, dims.data()};
  const auto datatype = H5::PredType::NATIVE_DOUBLE;
  datatype.setOrder(H5T_ORDER_LE);

  const auto dataset =
      file.createDataSet(dataset_name, datatype, dataspace);

  auto m = DO::Sara::Tensor<double, 2, DO::Sara::RowMajor>{M, N};
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      m.matrix()(i, j) = i * N + j;
  std::cout << m.matrix() << std::endl;

  dataset.write(m.data(), H5::PredType::NATIVE_DOUBLE);
}

void read_data()
{
  //DO::Sara::MatrixXd m = DO::Sara::MatrixXd::Zero(M, N);
  DO::Sara::Tensor<double, 2, DO::Sara::RowMajor> m{M, N};
  m.matrix().setZero();

  H5::Exception::dontPrint();

  auto file = H5::H5File{filepath, H5F_ACC_RDONLY};
  auto dataset = file.openDataSet(dataset_name);

  // Retrieve the data types (int, float, double...)?
  auto type_class = dataset.getTypeClass();
  //if (type_class != H5T_NATIVE_DOUBLE)
  //  throw std::runtime_error{"Data type must be double!"};
  auto float_type = dataset.getFloatType();

  // Retrieve the data endianness.
  auto order_string = H5std_string{};
  auto order = float_type.getOrder(order_string);
  std::cout << order_string << std::endl;

  // Number of elements.
  auto size = float_type.getSize();
  std::cout << "data size = " << size << std::endl;

  // Dataspace.
  auto dataspace = dataset.getSpace();
  auto rank = dataspace.getSimpleExtentNdims();

  auto dims_out = std::array<hsize_t, 2>{};
  auto ndims = dataspace.getSimpleExtentDims(dims_out.data(), nullptr);
  std::cout << "rank " << rank << ", dimensions "
            << static_cast<unsigned long>(dims_out[0]) << " x "
            << static_cast<unsigned long>(dims_out[1]) << std::endl;

  // Source data.
  auto offset = std::array<hsize_t, 2>{1, 2};
  auto count = std::array<hsize_t, 2>{2, 3};
  dataspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());

  // Destination data.
  auto dimsm = std::array<hsize_t, 2>{M, N};
  auto memspace = H5::DataSpace{2, dimsm.data()};
  memspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());

  dataset.read(m.data(), H5::PredType::NATIVE_DOUBLE, memspace, dataspace);

  std::cout << "single_matrix = " << std::endl << m.matrix() << std::endl;
}

int main()
{
  try
  {
    write_data();
    read_data();
  }
  catch (H5::FileIException& e)
  {
    e.printError();
    return -1;
  }
  catch (H5::DataSetIException& e)
  {
    e.printError();
    return -1;
  }
  catch (H5::DataSpaceIException& e)
  {
    e.printError();
    return -1;
  }
  catch (H5::DataTypeIException& e)
  {
    e.printError();
    return -1;
  }

  return 0;
}

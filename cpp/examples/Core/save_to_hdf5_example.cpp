#include <DO/Sara/Core.hpp>
#include <DO/Sara/Core/Tensor.hpp>

#if defined(__APPLE__)
# include <H5Cpp.h>
#else
# include <hdf5/serial/H5Cpp.h>
#endif

#include <array>
#include <memory>


constexpr auto filepath = "data.h5";
constexpr auto rank = 2;
constexpr auto M = 5;
constexpr auto N = 6;


void write_data()
{
  std::cout << "Write" << std::endl;

  H5::Exception::dontPrint();

  auto file = H5::H5File{filepath, H5F_ACC_TRUNC};

  // Store the model.
  {
    auto models_group =
        std::unique_ptr<H5::Group>(new H5::Group(file.createGroup("/models")));

    const auto dims = std::array<hsize_t, rank>{M, N};
    const auto dataspace = H5::DataSpace{rank, dims.data()};
    const auto datatype = H5::PredType::NATIVE_DOUBLE;
    datatype.setOrder(H5T_ORDER_LE);

    const auto dataset =
      file.createDataSet("/models/weights", datatype, dataspace);

    auto m = DO::Sara::Tensor<double, 2, DO::Sara::RowMajor>{M, N};
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        m.matrix()(i, j) = i * N + j;

    dataset.write(m.data(), H5::PredType::NATIVE_DOUBLE);
  }

  // Store the train data.
  {
    auto data_group = std::unique_ptr<H5::Group>(
        new H5::Group(file.createGroup("/data")));
    auto train_group = std::unique_ptr<H5::Group>(
        new H5::Group(file.createGroup("/data/train")));

    const auto x_dims = std::array<hsize_t, rank>{M, N};
    const auto x_dataspace = H5::DataSpace{rank, x_dims.data()};
    const auto x_datatype = H5::PredType::NATIVE_DOUBLE;
    x_datatype.setOrder(H5T_ORDER_LE);
    const auto x_dataset = file.createDataSet("/data/train/x", x_datatype, x_dataspace);

    const auto y_dims = std::array<hsize_t, rank>{1, N};
    const auto y_dataspace = H5::DataSpace{rank, y_dims.data()};
    const auto y_datatype = H5::PredType::NATIVE_INT;
    y_datatype.setOrder(H5T_ORDER_LE);
    const auto y_dataset = file.createDataSet("/data/train/y", y_datatype, y_dataspace);

    auto x = DO::Sara::Tensor<double, 2, DO::Sara::RowMajor>{M, N};
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        x.matrix()(i, j) = i * N + j;

    auto y = DO::Sara::Tensor<int, 2, DO::Sara::RowMajor>{1, N};
    for (int j = 0; j < N; ++j)
      y.matrix()(0, j) = j;

    x_dataset.write(x.data(), H5::PredType::NATIVE_DOUBLE);
    y_dataset.write(y.data(), H5::PredType::NATIVE_INT);
  }
}

void read_data()
{
  std::cout << "Read" << std::endl;
  DO::Sara::Tensor<double, 2, DO::Sara::RowMajor> m{M, N};
  m.matrix().setZero();

  H5::Exception::dontPrint();

  auto file = H5::H5File{filepath, H5F_ACC_RDONLY};
  auto group = H5::Group(file.openGroup("models"));
  auto dataset = H5::DataSet(group.openDataSet("weights"));

  // Retrieve the data types (int, float, double...)?
  auto type_class = dataset.getTypeClass();
  auto float_type = dataset.getFloatType();

  // Retrieve the data endianness.
  auto order_string = H5std_string{};
  auto order = float_type.getOrder(order_string);
  std::cout << "order_string = " << order_string << std::endl;
  std::cout << "order = " << order << std::endl;

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

  std::cout << "weights = " << std::endl << m.matrix() << std::endl;
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
    e.printErrorStack();
    return -1;
  }
  catch (H5::DataSetIException& e)
  {
    e.printErrorStack();
    return -1;
  }
  catch (H5::DataSpaceIException& e)
  {
    e.printErrorStack();
    return -1;
  }
  catch (H5::DataTypeIException& e)
  {
    e.printErrorStack();
    return -1;
  }

  return 0;
}

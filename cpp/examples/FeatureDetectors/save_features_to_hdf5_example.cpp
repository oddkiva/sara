#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Features/Feature.hpp>

#if defined(__APPLE__)
# include <H5Cpp.h>
#else
# include <hdf5/serial/H5Cpp.h>
#endif

#include <array>
#include <iostream>
#include <memory>


using namespace DO::Sara;


template <typename T>
struct h5_native_type;

template <>
struct h5_native_type<float>
{ static H5::PredType value() { return H5::PredType::NATIVE_FLOAT; } };

template <>
struct h5_native_type<double>
{ static H5::PredType value() { return H5::PredType::NATIVE_DOUBLE; } };

template <typename T, int M, int N>
constexpr auto calculate_h5_type(const Eigen::Matrix<T, M, N>&)
{
  const hsize_t dims[] = {M, N};
  return H5::ArrayType{h5_native_type<T>::value(), 2, dims};
}

template <typename T, int N>
constexpr auto calculate_h5_type(const Eigen::Matrix<T, N, 1>&)
{
  const hsize_t dims[] = {N};
  return H5::ArrayType{h5_native_type<T>::value(), 1, dims};
}

template <typename T, int N>
constexpr auto calculate_h5_type(const Eigen::Matrix<T, 1, N>&)
{
  const hsize_t dims[] = {N};
  return H5::ArrayType{h5_native_type<T>::value(), 1, dims};
}


template <typename T>
struct CalculateH5Type;

template <>
struct CalculateH5Type<float>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_FLOAT;
  };
};

template <>
struct CalculateH5Type<OERegion::Type>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_UINT8;
  };
};

template <>
struct CalculateH5Type<OERegion::ExtremumType>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_INT8;
  };
};

template <typename T, int M, int N>
struct CalculateH5Type<Matrix<T, M, N>>
{
  static inline auto value() -> H5::ArrayType
  {
    if constexpr (M == 1 || N == 1)
    {
      const hsize_t dims[] = {M * N};
      return {h5_native_type<T>::value(), 1, dims};
    }
    else
    {
      const hsize_t dims[] = {M, N};
      return {h5_native_type<T>::value(), 2, dims};
    }
  };
};

template <typename T>
inline auto calculate_h5_type()
{
  return CalculateH5Type<T>::value();
}


#define INSERT_MEMBER(comp_type, struct_t, member_name)                        \
  {                                                                            \
    using member_type = decltype(struct_t{}.member_name);                      \
    const auto member_h5_type = calculate_h5_type<member_type>();              \
    comp_type.insertMember(#member_name, HOFFSET(struct_t, member_name),       \
                           member_h5_type);                                    \
  }

template <>
struct CalculateH5Type<OERegion>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(OERegion)};
    INSERT_MEMBER(h5_comp_type, OERegion, coords);
    INSERT_MEMBER(h5_comp_type, OERegion, shape_matrix);
    INSERT_MEMBER(h5_comp_type, OERegion, orientation);
    INSERT_MEMBER(h5_comp_type, OERegion, extremum_value);
    INSERT_MEMBER(h5_comp_type, OERegion, type);
    INSERT_MEMBER(h5_comp_type, OERegion, extremum_type);
    return h5_comp_type;
  }
};



auto create_h5_file(const std::string& filename,
                    unsigned int flags = H5F_ACC_TRUNC)
{
  return std::shared_ptr<H5::H5File>{new H5::H5File{filename, flags}};
}


struct H5File
{
  H5File() = default;

  explicit H5File(const std::string& filename, unsigned int flags)
    : file{new H5::H5File{filename, flags}}
  {
  }

  auto group(const std::string& group_name)
  {
    auto group_it = groups.find(group_name);
    if (group_it != groups.end())
      return group_it->second;

    auto group = std::shared_ptr<H5::Group>{
        new H5::Group{file->createGroup(group_name)}};
    groups[group_name] = group;
    return group;
  }

  template <typename T, int Rank>
  auto write_dataset(const std::string& dataset_name,
                     const DO::Sara::Tensor_<T, Rank>& data)
  {
    const auto data_type = CalculateH5Type<T>::value();

    auto data_dims = std::array<hsize_t, Rank>{};
    std::transform(data.sizes().data(),
                   data.sizes().data() + data.sizes().size(), data_dims.data(),
                   [](auto val) { return hsize_t(val); });
    const auto data_space = H5::DataSpace{Rank, data_dims.data()};

    auto dataset = file->createDataSet(dataset_name, data_type, data_space);
    dataset.write(data.data(), data_type);

    return dataset;
  }

  template <typename T, int Rank>
  auto read_dataset(const std::string& dataset_name, Tensor_<T, Rank>& data)
  {
    using vector_type = Eigen::Matrix<hsize_t, Rank, 1>;

    auto dataset = H5::DataSet(file->openDataSet(dataset_name));

    // // Retrieve the data types (int, float, double...)?
    // auto type_class = dataset.getTypeClass();
    // auto float_type = dataset.getFloatType();

    // // Retrieve the data endianness.
    // auto order_string = H5std_string{};
    // auto order = float_type.getOrder(order_string);
    // std::cout << "order_string = " << order_string << std::endl;
    // std::cout << "order = " << order << std::endl;

    // // Number of elements in bytes.
    // auto byte_size = float_type.getSize();
    // std::cout << "file data byte size = " << byte_size << std::endl;

    // File data space.
    auto file_data_space = dataset.getSpace();
    auto file_data_rank = file_data_space.getSimpleExtentNdims();
    std::cout << "file data rank = " << file_data_rank << std::endl;

    auto file_data_dims = vector_type{};
    file_data_space.getSimpleExtentDims(file_data_dims.data(), nullptr);
    std::cout << "file data dims = " << file_data_dims.transpose() << std::endl;

    // Select the portion of the file data.
    const vector_type file_offset = vector_type::Zero();
    const auto file_count = file_data_dims;
    file_data_space.selectHyperslab(H5S_SELECT_SET, file_count.data(), file_offset.data());
    std::cout << "Selected src offset = " << file_offset.transpose() << std::endl;
    std::cout << "Selected src count = " << file_count.transpose() << std::endl;

    // Select the portion of the destination data.
    const vector_type dst_offset = vector_type::Zero();
    const auto dst_count = file_data_dims;
    const auto dst_space = H5::DataSpace{Rank, dst_count.data()};
    dst_space.selectHyperslab(H5S_SELECT_SET, dst_count.data(), dst_offset.data());

    // Resize the data.
    auto data_sizes = typename DO::Sara::Tensor_<T, Rank>::vector_type{};
    for (int i = 0; i < Rank; ++i)
      data_sizes[i] = dst_count[i];
    data.resize(data_sizes);

    // Read the data.
    dataset.read(data.data(), calculate_h5_type<T>(), dst_space, file_data_space);
  }

  std::map<std::string, std::shared_ptr<H5::Group>> groups;
  std::shared_ptr<H5::H5File> file;
};


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

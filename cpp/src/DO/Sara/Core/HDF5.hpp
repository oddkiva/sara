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

#pragma once

#include <DO/Sara/Core/DebugUtilities.hpp>
#include <DO/Sara/Core/Tensor.hpp>
#include <DO/Sara/Features/Feature.hpp>

#if defined(__APPLE__) || defined(_WIN32)
# include <H5Cpp.h>
#else
# include <hdf5/serial/H5Cpp.h>
#endif

#include <array>
#include <iostream>
#include <memory>


namespace DO::Sara {

template <typename T>
struct CalculateH5Type;


template <typename T>
inline auto calculate_h5_type()
{
  return CalculateH5Type<T>::value();
}

template <>
struct CalculateH5Type<unsigned short>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_USHORT;
  };
};

template <>
struct CalculateH5Type<unsigned int>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_UINT;
  };
};

template <>
struct CalculateH5Type<int>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_INT;
  };
};

template <>
struct CalculateH5Type<float>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_FLOAT;
  };
};

template <>
struct CalculateH5Type<double>
{
  static inline auto value()
  {
    return H5::PredType::NATIVE_DOUBLE;
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
      return {calculate_h5_type<T>(), 1, dims};
    }
    else
    {
      const hsize_t dims[] = {M, N};
      return {calculate_h5_type<T>(), 2, dims};
    }
  };
};


#define INSERT_MEMBER(comp_type, struct_t, member_name)                        \
  {                                                                            \
    using member_type = decltype(std::declval<struct_t>().member_name);        \
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

} /* namespace DO::Sara */


namespace DO::Sara {

struct H5File
{
  using vector_type = Matrix<hsize_t, Dynamic, 1>;

  template <int Rank>
  using fixed_vector_type = Eigen::Matrix<hsize_t, Rank, 1>;


  H5File() = default;

  explicit H5File(const std::string& filename, unsigned int flags)
    : file{new H5::H5File{filename, flags}}
  {
  }

  ~H5File()
  {
    file->close();
  }

  auto find_group(const std::string& group_name)
  {
    auto group = std::unique_ptr<H5::Group>{};
    if (H5Lexists(file->getId(), group_name.c_str(), H5P_DEFAULT) > 0)
      group.reset(new H5::Group{file->openGroup(group_name)});
    return group;
  }

  auto get_group(const std::string& group_name)
  {
    auto group = find_group(group_name);
    if (group == nullptr)
      group.reset(new H5::Group{file->createGroup(group_name)});
    return group;
  }

  auto find_dataset(const std::string& dataset_name)
  {
    auto dataset = std::unique_ptr<H5::DataSet>{};
    if (H5Lexists(file->getId(), dataset_name.c_str(), H5P_DEFAULT) > 0)
    {
      SARA_DEBUG << "Find dataset " << dataset_name << "..." << std::endl;
      dataset.reset(new H5::DataSet{file->openDataSet(dataset_name)});
    }
    return dataset;
  }

  auto read_dataset_sizes(const H5::DataSpace& file_data_space)
  {
    // File data space.
    auto file_data_rank = file_data_space.getSimpleExtentNdims();
#ifdef DEBUG
    SARA_DEBUG << "file data rank = " << file_data_rank << std::endl;
#endif

    auto file_data_dims = vector_type(file_data_rank);
    file_data_space.getSimpleExtentDims(file_data_dims.data(), nullptr);
#ifdef DEBUG
    SARA_DEBUG << "file data dims = " << file_data_dims.transpose()
               << std::endl;
#endif
    return file_data_dims;
  }

  auto read_dataset_sizes(const H5::DataSet& dataset)
  {
    return read_dataset_sizes(dataset.getSpace());
  }

  template <typename T, int Rank>
  auto write_dataset(const std::string& dataset_name,
                     const DO::Sara::TensorView_<T, Rank>& data,
                     bool overwrite = false)
  {
    const auto data_type = CalculateH5Type<T>::value();

    auto data_dims = std::array<hsize_t, Rank>{};
    std::transform(data.sizes().data(),
                   data.sizes().data() + data.sizes().size(), data_dims.data(),
                   [](auto val) { return hsize_t(val); });
    const auto data_space = H5::DataSpace{Rank, data_dims.data()};

    auto dataset = find_dataset(dataset_name);
    if (dataset != nullptr && overwrite)
      delete_dataset(dataset_name);

    if (!overwrite && dataset != nullptr)
      throw std::runtime_error{"Error: dataset \"" + dataset_name +
                               "\" exists but overwriting is not permitted!"};

    dataset.reset(new H5::DataSet{
        file->createDataSet(dataset_name, data_type, data_space)});
    dataset->write(data.data(), data_type);

    return dataset;
  }

  template <typename T>
  auto read_dataset(const std::string& dataset_name, std::vector<T>& data)
  {
    auto dataset = H5::DataSet(file->openDataSet(dataset_name));

    // File data space.
    auto file_data_space = dataset.getSpace();
    const auto file_data_dims = read_dataset_sizes(file_data_space);
    const auto file_data_rank = int(file_data_dims.size());

    // Select the portion of the file data.
    const vector_type file_offset = vector_type::Zero(file_data_rank);
    const auto file_count = file_data_dims;
    file_data_space.selectHyperslab(H5S_SELECT_SET, file_count.data(),
                                    file_offset.data());
#ifdef DEBUG
    SARA_DEBUG << "Selected src offset = " << file_offset << std::endl;
    SARA_DEBUG << "Selected src count = " << file_count << std::endl;
#endif

    // Select the portion of the destination data.
    const vector_type dst_offset = vector_type::Zero(file_data_rank);
    const auto dst_count = file_data_dims;
    const auto dst_space = H5::DataSpace{file_data_rank, dst_count.data()};
    dst_space.selectHyperslab(H5S_SELECT_SET, dst_count.data(),
                              dst_offset.data());

    // Resize the data.
    const auto data_sizes = dst_count.template cast<size_t>().eval();
    const auto data_flat_size = std::accumulate(
        data_sizes.data(), data_sizes.data() + data_sizes.size(), size_t(1),
        std::multiplies<size_t>());
    data.resize(data_flat_size);

    // Read the data.
    dataset.read(data.data(), calculate_h5_type<T>(), dst_space,
                 file_data_space);
  }

  template <typename T, int Rank>
  auto read_dataset(const std::string& dataset_name, Tensor_<T, Rank>& data)
  {
    auto dataset = H5::DataSet(file->openDataSet(dataset_name));

    // File data space.
    auto file_data_space = dataset.getSpace();
#ifdef DEBUG
    auto file_data_rank = file_data_space.getSimpleExtentNdims();
    SARA_DEBUG << "file data rank = " << file_data_rank << std::endl;
#endif

    auto file_data_dims = fixed_vector_type<Rank>{};
    file_data_space.getSimpleExtentDims(file_data_dims.data(), nullptr);
#ifdef DEBUG
    SARA_DEBUG << "file data dims = " << file_data_dims.transpose()
               << std::endl;
#endif

    // Select the portion of the file data.
    const fixed_vector_type<Rank> file_offset = fixed_vector_type<Rank>::Zero();
    const auto file_count = file_data_dims;
    file_data_space.selectHyperslab(H5S_SELECT_SET, file_count.data(),
                                    file_offset.data());
#ifdef DEBUG
    SARA_DEBUG << "Selected src offset = " << file_offset.transpose()
               << std::endl;
    SARA_DEBUG << "Selected src count = " << file_count.transpose() << std::endl;
#endif

    // Select the portion of the destination data.
    const fixed_vector_type<Rank> dst_offset = fixed_vector_type<Rank>::Zero();
    const auto dst_count = file_data_dims;
    const auto dst_space = H5::DataSpace{Rank, dst_count.data()};
    dst_space.selectHyperslab(H5S_SELECT_SET, dst_count.data(),
                              dst_offset.data());

    // Resize the data.
    data.resize(dst_count.template cast<int>().eval());

    // Read the data.
    dataset.read(data.data(), calculate_h5_type<T>(), dst_space,
                 file_data_space);
  }

  auto delete_dataset(const std::string& dataset_name) -> void
  {
    int result = H5Ldelete(file->getId(), dataset_name.c_str(), H5P_DEFAULT);
    if (result < 0)
      throw std::runtime_error{"Error: could not delete dataset: " +
                               dataset_name};
  }

  std::shared_ptr<H5::H5File> file;
};

} /* namespace DO::Sara */

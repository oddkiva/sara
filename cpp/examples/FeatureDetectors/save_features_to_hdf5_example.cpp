#include <DO/Sara/Features/Feature.hpp>

#if defined(__APPLE__)
# include <H5Cpp.h>
#else
# include <hdf5/serial/H5Cpp.h>
#endif

#include <array>
#include <iostream>


struct feature_t
{
  DO::Sara::Vector2f pos;
  DO::Sara::Matrix2f scale_matrix;
};

int main()
{
  const auto filename = H5std_string{"features.h5"};
  const auto dataset_name = H5std_string{"features.h5"};
  const auto type =  H5std_string{"type"};

  const hsize_t dim[] = {4};
  constexpr auto rank = 1;
  const H5::DataSpace space(rank, dim);

  auto file =
      std::unique_ptr<H5::H5File>(new H5::H5File{filename, H5F_ACC_TRUNC});

  auto features = std::array<feature_t, 4>{};
  for (int i = 0; i < 4; ++i)
  {
    features[i].pos << i, i;
    features[i].scale_matrix = DO::Sara::Matrix2f::Ones() * (i + 0.5f);
  }

  H5::CompType feature_t_type{sizeof(feature_t)};

  const hsize_t pos_dim[] = {2};
  const auto pos_array = H5::ArrayType{H5::PredType::NATIVE_FLOAT, 1, pos_dim};
  feature_t_type.insertMember("pos", HOFFSET(feature_t, pos), pos_array);

  const hsize_t scale_dim[] = {2, 2};
  const auto scale_array = H5::ArrayType{H5::PredType::NATIVE_FLOAT, 2, scale_dim};
  feature_t_type.insertMember("scale_matrix", HOFFSET(feature_t, scale_matrix), scale_array);

  auto dataset = std::unique_ptr<H5::DataSet>{new H5::DataSet{
      file->createDataSet(dataset_name, feature_t_type, space)}};

  dataset->write(features.data(), feature_t_type);
  std::cout << "WRITE OK" << std::endl;

  auto features2 = std::array<feature_t, 4>{};
  dataset->read(features2.data(), feature_t_type);
  std::cout << "READ OK" << std::endl;

  for (int i = 0; i < 4; ++i)
  {
    std::cout << "index " << i << std::endl;
    std::cout << "pos = \n" << features[i].pos << std::endl;
    std::cout << "scale_matrix = \n" << features[i].scale_matrix << std::endl;
  }

  return 0;
}

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

#include <DO/Sara/Core/CSV.hpp>
#include <DO/Sara/Core/HDF5.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO/Details/Exif.hpp>

#include <boost/filesystem.hpp>

#include <iterator>


namespace fs = boost::filesystem;
namespace sara = DO::Sara;


namespace DO::Sara {

//! @brief Minimalistic EXIF metadata easy to serialize with HDF5.
struct ExifMetadata
{
  // Focal length of lens in millimeters.
  double focal_length{0.};
  // Focal length in 35mm film.
  unsigned short focal_length_in_35_mm{0};

  // Image width reported in EXIF data
  unsigned image_width{0};
  // Image height reported in EXIF data
  unsigned image_height{0};

  // EXIFInfo::Geolocation_t geo_location{
  //    0., 0., 0., 0, {0., 0., 0., 0}, {0., 0., 0., 0}};

  ExifMetadata() = default;

  ExifMetadata(const ::EXIFInfo& e)
    : focal_length{e.FocalLength}
    , focal_length_in_35_mm{e.FocalLengthIn35mm}
    , image_width{e.ImageWidth}
    , image_height{e.ImageHeight}
  //, geo_location{e.GeoLocation}
  {
  }
};

template <>
struct CalculateH5Type<ExifMetadata>
{
  static inline auto value() -> H5::CompType
  {
    auto h5_comp_type = H5::CompType{sizeof(ExifMetadata)};
    INSERT_MEMBER(h5_comp_type, ExifMetadata, focal_length);
    INSERT_MEMBER(h5_comp_type, ExifMetadata, focal_length_in_35_mm);
    INSERT_MEMBER(h5_comp_type, ExifMetadata, image_width);
    INSERT_MEMBER(h5_comp_type, ExifMetadata, image_height);
    return h5_comp_type;
  }
};

} /* namespace DO::Sara */


void extract_exif()
{
  const auto dirpath = fs::path{
      "/mnt/a1cc5981-3655-4f74-9c62-37253d79c82d/sfm/Trafalgar/images"};
  const auto image_paths = sara::ls(dirpath.string(), ".jpg");

  to_csv(image_paths, "/home/david/Desktop/Datasets/Trafalgar.csv");

  auto exif_data = std::vector<sara::ExifMetadata>{};
  exif_data.reserve(image_paths.size());

  std::transform(std::begin(image_paths), std::end(image_paths),
                 std::back_inserter(exif_data),
                 [&](const auto& path) -> sara::ExifMetadata {
                   SARA_DEBUG << "Reading exif data from image " << path
                              << "..." << std::endl;
                   auto exif_info = EXIFInfo{};
                   sara::read_exif_info(exif_info, path);

                   SARA_DEBUG << "EXIF DATA:\n" << exif_info << std::endl;
                   SARA_DEBUG
                       << "shutter speed value: " << exif_info.ShutterSpeedValue
                       << std::endl;

                   return exif_info;
                 });

  SARA_CHECK(exif_data.size());

  auto h5_file =
      sara::H5File{"/home/david/Desktop/Datasets/Trafalgar.h5", H5F_ACC_TRUNC};
  auto exif_view = tensor_view(exif_data);
  h5_file.write_dataset("exif_metadata", exif_view);
}

void read_exif()
{
  const auto dirpath = fs::path{
      "/mnt/a1cc5981-3655-4f74-9c62-37253d79c82d/sfm/Trafalgar/images"};

  auto exif_data = sara::Tensor_<sara::ExifMetadata, 1>{};

  SARA_CHECK(exif_data.size());

  auto h5_file =
      sara::H5File{"/home/david/Desktop/Datasets/Trafalgar.h5", H5F_ACC_RDONLY};

  h5_file.read_dataset("exif_metadata", exif_data);

  for (const auto& exif : exif_data)
    SARA_DEBUG << "fl = " << exif.focal_length
               << " fl35: " << exif.focal_length_in_35_mm
               << " w: " << exif.image_width << " h: " << exif.image_width
               << std::endl;
}

GRAPHICS_MAIN()
{
  extract_exif();
  read_exif();
  return 0;
}

// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2017 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/ImageIO/Details/Exif.hpp>

#include <fmt/format.h>

#include <vector>


std::ostream& operator<<(std::ostream& os, const EXIFInfo& info)
{
  std::string exif_info_string;

  exif_info_string += fmt::format("Camera make       : {}\n", info.Make);
  exif_info_string += fmt::format("Camera model      : {}\n", info.Model);
  exif_info_string += fmt::format("Software          : {}\n", info.Software);

  // Image format and geometry metadata.
  exif_info_string +=
      fmt::format("Bits per sample   : {}\n", info.BitsPerSample);
  exif_info_string += fmt::format("Image width       : {}\n", info.ImageWidth);
  exif_info_string += fmt::format("Image height      : {}\n", info.ImageHeight);
  exif_info_string += fmt::format("Image orientation : {}\n", info.Orientation);

  exif_info_string +=
      fmt::format("Image description : {}\n", info.ImageDescription);
  exif_info_string += fmt::format("Image copyright   : {}\n", info.Copyright);

  // Time metadata.
  exif_info_string += fmt::format("Image datetime    : {}\n", info.DateTime);
  exif_info_string +=
      fmt::format("Original datetime : {}\n", info.DateTimeOriginal);
  exif_info_string +=
      fmt::format("Digitize datetime : {}\n", info.DateTimeDigitized);
  exif_info_string +=
      fmt::format("Subsecond time    : {}\n", info.SubSecTimeOriginal);
  exif_info_string +=
      fmt::format("Exposure time     : 1/{} s\n",
                  static_cast<unsigned>(1.0 / info.ExposureTime));

  // Camera lens parameters.
  exif_info_string +=
      fmt::format("F-stop            : f/{:0.1f}\n", info.FNumber);
  exif_info_string +=
      fmt::format("ISO speed         : {}\n", info.ISOSpeedRatings);
  exif_info_string +=
      fmt::format("Subject distance  : {} m\n", info.SubjectDistance);
  exif_info_string +=
      fmt::format("Exposure bias     : {} EV\n", info.ExposureBiasValue);
  exif_info_string +=
      fmt::format("Flash used?       : {}\n", static_cast<int>(info.Flash));
  exif_info_string +=
      fmt::format("Metering mode     : {}\n", info.MeteringMode);
  exif_info_string +=
      fmt::format("Lens focal length : {} mm\n", info.FocalLength);
  exif_info_string +=
      fmt::format("35mm focal length : {} mm\n", info.FocalLengthIn35mm);

  // GPS metadata.
  exif_info_string += fmt::format(
      "GPS Latitude      : {} deg ({} deg, {} min, {} sec {})\n",
      info.GeoLocation.Latitude, info.GeoLocation.LatComponents.degrees,
      info.GeoLocation.LatComponents.minutes,
      info.GeoLocation.LatComponents.seconds,
      info.GeoLocation.LatComponents.direction);
  exif_info_string += fmt::format(
      "GPS Longitude     : {} deg ({} deg, {} min, {} sec {})\n",
      info.GeoLocation.Longitude, info.GeoLocation.LonComponents.degrees,
      info.GeoLocation.LonComponents.minutes,
      info.GeoLocation.LonComponents.seconds,
      info.GeoLocation.LonComponents.direction);
  exif_info_string +=
      fmt::format("GPS Altitude      : {} m\n", info.GeoLocation.Altitude);

  os << exif_info_string;
  return os;
}


namespace DO::Sara {

  auto read_exif_info(EXIFInfo& info, const std::string& file_path) -> bool
  {
    // Read the JPEG file into a buffer
    const auto fp = fopen(file_path.c_str(), "rb");
    if (!fp)
    {
      std::cout << "Can't open file:" << std::endl << file_path << std::endl;
      return false;
    }
    fseek(fp, 0, SEEK_END);
    const auto fsize = static_cast<unsigned long>(ftell(fp));
    rewind(fp);

    std::vector<unsigned char> buf(fsize);
    if (fread(&buf[0], 1, fsize, fp) != fsize)
    {
      std::cout << "Can't read file: " << std::endl << file_path << std::endl;
      return false;
    }
    fclose(fp);

    // Parse EXIF info.
    const auto code = info.parseFrom(&buf[0], static_cast<unsigned int>(fsize));

    return !code;
  }

}  // namespace DO::Sara

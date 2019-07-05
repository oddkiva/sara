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

#include <vector>

#include <DO/Sara/ImageIO/Details/Exif.hpp>


using namespace std;


ostream& operator<<(ostream& os, const EXIFInfo& exifInfo)
{
  vector<char> buffer(1000);
  int length;
  string exif_info_string;

  length =
      sprintf(&buffer[0], "Camera make       : %s\n", exifInfo.Make.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "Camera model      : %s\n", exifInfo.Model.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Software          : %s\n",
                   exifInfo.Software.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "Bits per sample   : %d\n", exifInfo.BitsPerSample);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Image width       : %d\n", exifInfo.ImageWidth);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "Image height      : %d\n", exifInfo.ImageHeight);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Image description : %s\n",
                   exifInfo.ImageDescription.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "Image orientation : %d\n", exifInfo.Orientation);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Image copyright   : %s\n",
                   exifInfo.Copyright.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Image date/time   : %s\n",
                   exifInfo.DateTime.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Original date/time: %s\n",
                   exifInfo.DateTimeOriginal.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Digitize date/time: %s\n",
                   exifInfo.DateTimeDigitized.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Subsecond time    : %s\n",
                   exifInfo.SubSecTimeOriginal.c_str());
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Exposure time     : 1/%d s\n",
                   (unsigned) (1.0 / exifInfo.ExposureTime));
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "F-stop            : f/%.1f\n", exifInfo.FNumber);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "ISO speed         : %d\n", exifInfo.ISOSpeedRatings);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Subject distance  : %f m\n",
                   exifInfo.SubjectDistance);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Exposure bias     : %f EV\n",
                   exifInfo.ExposureBiasValue);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "Flash used?       : %d\n", exifInfo.Flash);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "Metering mode     : %d\n", exifInfo.MeteringMode);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length =
      sprintf(&buffer[0], "Lens focal length : %f mm\n", exifInfo.FocalLength);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0], "35mm focal length : %u mm\n",
                   exifInfo.FocalLengthIn35mm);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(
      &buffer[0], "GPS Latitude      : %f deg (%f deg, %f min, %f sec %c)\n",
      exifInfo.GeoLocation.Latitude, exifInfo.GeoLocation.LatComponents.degrees,
      exifInfo.GeoLocation.LatComponents.minutes,
      exifInfo.GeoLocation.LatComponents.seconds,
      exifInfo.GeoLocation.LatComponents.direction);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  length = sprintf(&buffer[0],
                   "GPS Longitude     : %f deg (%f deg, %f min, %f sec %c)\n",
                   exifInfo.GeoLocation.Longitude,
                   exifInfo.GeoLocation.LonComponents.degrees,
                   exifInfo.GeoLocation.LonComponents.minutes,
                   exifInfo.GeoLocation.LonComponents.seconds,
                   exifInfo.GeoLocation.LonComponents.direction);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  // GPS altitude.
  length = sprintf(&buffer[0], "GPS Altitude      : %f m\n",
                   exifInfo.GeoLocation.Altitude);
  exif_info_string += string(&buffer[0], &buffer[0] + length);

  os << exif_info_string;
  return os;
}


namespace DO { namespace Sara {

  bool read_exif_info(EXIFInfo& exif_info, const std::string& file_path)
  {
    // Read the JPEG file into a buffer
    FILE *fp = fopen(file_path.c_str(), "rb");
    if (!fp)
    {
      cout << "Can't open file:" << endl << file_path << endl;
      return false;
    }
    fseek(fp, 0, SEEK_END);
    unsigned long fsize = ftell(fp);
    rewind(fp);

    vector<unsigned char> buf(fsize);
    if (fread(&buf[0], 1, fsize, fp) != fsize)
    {
      cout << "Can't read file: " << endl << file_path << endl;
      return false;
    }
    fclose(fp);

    // Parse EXIF info.
    int code = exif_info.parseFrom(&buf[0], fsize);

    return !code;
  }

} /* namespace Sara */
} /* namespace DO */

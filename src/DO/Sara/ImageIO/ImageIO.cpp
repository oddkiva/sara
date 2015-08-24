// ========================================================================== //
// This file is part of DO-CV, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //


#include <vector>

#if defined(_WIN32) || defined(_WIN32_WCE)
# define NOMINMAX
# include <windows.h>
#endif

#include <DO/Sara/ImageIO/ImageIO.hpp>
#include <DO/Sara/Core/Image.hpp>
#include <DO/Sara/ImageIO/ImageIOObjects.hpp>


using namespace std;


// Utilities.
namespace DO { namespace Sara {

  static inline string file_ext(const string& filepath)
  {
    if (filepath.empty())
      return string{};

    string ext{ filepath.substr(filepath.find_last_of(".")) };
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
  }

  static inline bool is_jpeg_file_ext(const string& ext)
  {
    return
      ext == ".jpg"  ||
      ext == ".jpeg" ||
      ext == ".jpe"  ||
      ext == ".jfif" ||
      ext == ".jfi";
  }

  static inline bool is_png_file_ext(const string& ext)
  {
    return ext == ".png";
  }

  static inline bool is_tiff_file_ext(const string& ext)
  {
    return
      ext == ".tif" ||
      ext == ".tiff";
  }

} /* namespace Sara */
} /* namespace DO */


// Parsing of EXIF info.
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

  std::ostream& operator<<(std::ostream& os, const EXIFInfo& exifInfo)
  {
    vector<char> buffer(1000);
    int length;
    string exif_info_string;

    length = sprintf(&buffer[0], "Camera make       : %s\n", exifInfo.Make.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Camera model      : %s\n", exifInfo.Model.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Software          : %s\n", exifInfo.Software.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Bits per sample   : %d\n", exifInfo.BitsPerSample);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Image width       : %d\n", exifInfo.ImageWidth);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Image height      : %d\n", exifInfo.ImageHeight);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Image description : %s\n", exifInfo.ImageDescription.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Image orientation : %d\n", exifInfo.Orientation);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Image copyright   : %s\n", exifInfo.Copyright.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Image date/time   : %s\n", exifInfo.DateTime.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Original date/time: %s\n", exifInfo.DateTimeOriginal.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Digitize date/time: %s\n", exifInfo.DateTimeDigitized.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Subsecond time    : %s\n", exifInfo.SubSecTimeOriginal.c_str());
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Exposure time     : 1/%d s\n", (unsigned) (1.0/exifInfo.ExposureTime));
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "F-stop            : f/%.1f\n", exifInfo.FNumber);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "ISO speed         : %d\n", exifInfo.ISOSpeedRatings);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Subject distance  : %f m\n", exifInfo.SubjectDistance);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Exposure bias     : %f EV\n", exifInfo.ExposureBiasValue);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Flash used?       : %d\n", exifInfo.Flash);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Metering mode     : %d\n", exifInfo.MeteringMode);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "Lens focal length : %f mm\n", exifInfo.FocalLength);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "35mm focal length : %u mm\n", exifInfo.FocalLengthIn35mm);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "GPS Latitude      : %f deg (%f deg, %f min, %f sec %c)\n",
      exifInfo.GeoLocation.Latitude,
      exifInfo.GeoLocation.LatComponents.degrees,
      exifInfo.GeoLocation.LatComponents.minutes,
      exifInfo.GeoLocation.LatComponents.seconds,
      exifInfo.GeoLocation.LatComponents.direction);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    length = sprintf(&buffer[0], "GPS Longitude     : %f deg (%f deg, %f min, %f sec %c)\n",
      exifInfo.GeoLocation.Longitude,
      exifInfo.GeoLocation.LonComponents.degrees,
      exifInfo.GeoLocation.LonComponents.minutes,
      exifInfo.GeoLocation.LonComponents.seconds,
      exifInfo.GeoLocation.LonComponents.direction);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    // GPS altitude.
    length = sprintf(&buffer[0], "GPS Altitude      : %f m\n", exifInfo.GeoLocation.Altitude);
    exif_info_string += string(&buffer[0], &buffer[0]+length);

    os << exif_info_string;
    return os;
  }

} /* namespace Sara */
} /* namespace DO */


// Image read/write.
namespace DO { namespace Sara {

  static
  bool imread(unsigned char *& data, int& w, int& h, int& d,
              const string& filepath)
  {
    data = 0;
    w = h = d = 0;

    const auto ext = file_ext(filepath);

    if ( is_jpeg_file_ext(ext) &&
         JpegFileReader(filepath).read(data, w, h, d) )
      return true;
    if ( is_png_file_ext(ext) &&
         PngFileReader(filepath).read(data, w, h, d) )
      return true;
    if ( is_tiff_file_ext(ext) &&
         TiffFileReader(filepath).read(data, w, h, d) )
      return true;

    cerr << "Image format: " << ext << " either currently unsupported or invalid" << endl;
    return false;
  }

  bool imread(Image<unsigned char>& image, const std::string& filepath)
  {
    unsigned char *data;
    int w, h, d;

    if (!imread(data, w, h, d, filepath))
      return false;

    // Wrap data and get data ownership
    if (d == 1)
      image = Image<unsigned char>(&data[0], Vector2i(w,h));
    if (d == 3)
      image = Image<Rgb8>(reinterpret_cast<Rgb8 *>(&data[0]), Vector2i(w,h))
        .convert<unsigned char>();
    if (d == 4)
      image = Image<Rgba8>(reinterpret_cast<Rgba8 *>(&data[0]), Vector2i(w,h))
        .convert<unsigned char>();

    EXIFInfo info;
    if (read_exif_info(info, filepath))
      flip(image, int(info.Orientation));

    return true;
  }

  bool imread(Image<Rgb8>& image, const std::string& filepath)
  {
    unsigned char *data;
    int w, h, d;

    if (!imread(data, w, h, d, filepath))
      return false;

    // Wrap data and acquire data ownership.
    if (d == 1)
      image = Image<unsigned char>(&data[0], Vector2i(w,h)).convert<Rgb8>();
    else if (d == 3)
      image = Image<Rgb8>(reinterpret_cast<Rgb8 *>(&data[0]), Vector2i(w,h));
    else if (d == 4)
      image = Image<Rgba8>(reinterpret_cast<Rgba8 *>(&data[0]),
                           Vector2i(w,h)).convert<Rgb8>();

    EXIFInfo info;
    if (read_exif_info(info, filepath))
      flip(image, int(info.Orientation));
    return true;
  }

  bool imwrite(const Image<Rgb8>& image, const std::string& filepath,
               int quality)
  {
    string ext(file_ext(filepath));

    if (is_jpeg_file_ext(ext))
    {
      JpegFileWriter jpeg_writer(
        reinterpret_cast<const unsigned char *>(image.data()),
        image.width(), image.height(), 3);
      return jpeg_writer.write(filepath, quality);
    }

    if (is_png_file_ext(ext))
    {
      PngFileWriter png_writer(
        reinterpret_cast<const unsigned char *>(image.data()),
        image.width(), image.height(), 3);
      return png_writer.write(filepath, quality);
    }

    if (is_tiff_file_ext(ext))
    {
      TiffFileWriter tiff_writer(
        reinterpret_cast<const unsigned char *>(image.data()),
        image.width(), image.height(), 3);
      return tiff_writer.write(filepath, quality);
    }

    cout << ext << "is not a valid extension" << endl;
    return false;
  }

} /* namespace Sara */
} /* namespace DO */

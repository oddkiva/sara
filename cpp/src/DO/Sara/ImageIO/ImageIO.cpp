// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
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

#include <DO/Sara/Core/Image.hpp>

#include <DO/Sara/ImageIO/ImageIO.hpp>
#include <DO/Sara/ImageIO/Details/Exif.hpp>
#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>


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

    auto info = EXIFInfo{};
    if (read_exif_info(info, filepath))
      make_upright_from_exif(image, info.Orientation);

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

    auto info = EXIFInfo{};
    if (read_exif_info(info, filepath))
      make_upright_from_exif(image, info.Orientation);
    return true;
  }

  bool imwrite(const Image<Rgb8>& image, const std::string& filepath,
               int quality)
  {
    const auto ext = file_ext(filepath);

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

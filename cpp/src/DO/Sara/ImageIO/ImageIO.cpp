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
#include <DO/Sara/Core/StringFormat.hpp>

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

  template <typename ImageFileReader>
  void read_image_with(Image<unsigned char>& image, const char *filepath)
  {
    ImageFileReader reader{filepath};

    int w, h, d;
    std::tie(w, h, d) = reader.image_sizes();
    image.resize(w, h);

    if (d == 1)
    {
      reader.read(image.data());
    }
    else if (d == 3)
    {
      auto tmp = Image<Rgb8>{ w, h };
      reader.read(reinterpret_cast<unsigned char *>(tmp.data()));
      image = tmp.convert<unsigned char>();
    }
    else if (d == 4)
    {
      auto tmp = Image<Rgba8>{w, h};
      reader.read(reinterpret_cast<unsigned char *>(tmp.data()));
      image = tmp.convert<unsigned char>();
      return;
    }
    else
      throw std::runtime_error{
          "Unsupported number of input components in image file read!"};
  }

  template <typename ImageFileReader>
  void read_image_with(Image<Rgb8>& image, const char *filepath)
  {
    ImageFileReader reader{filepath};

    int w, h, d;
    tie(w, h, d) = reader.image_sizes();
    image.resize(w, h);

    if (d == 1)
    {
      auto tmp = Image<unsigned char>{w, h};
      reader.read(reinterpret_cast<unsigned char*>(tmp.data()));
      image = tmp.convert<Rgb8>();
    }
    else if (d == 3)
    {
      reader.read(reinterpret_cast<unsigned char*>(image.data()));
    }
    else if (d == 4)
    {
      auto tmp = Image<Rgba8>{w, h};
      reader.read(reinterpret_cast<unsigned char*>(tmp.data()));
      image = tmp.convert<Rgb8>();
    }
    else
      throw std::runtime_error{
          "Unsupported number of input components in image file!"};
  }

  void imread(Image<unsigned char>& image, const std::string& filepath)
  {
    const auto ext = file_ext(filepath);

    if (is_jpeg_file_ext(ext))
      read_image_with<JpegFileReader>(image, filepath.c_str());
    else if (is_png_file_ext(ext))
      read_image_with<PngFileReader>(image, filepath.c_str());
    else if (is_tiff_file_ext(ext))
      read_image_with<TiffFileReader>(image, filepath.c_str());
    else
      throw std::runtime_error{
          format("Image format: %s is either unsupported or invalid",
                 ext.c_str())
              .c_str()};

    auto info = EXIFInfo{};
    if (read_exif_info(info, filepath))
      make_upright_from_exif(image, info.Orientation);
  }

  void imread(Image<Rgb8>& image, const std::string& filepath)
  {
    const auto ext = file_ext(filepath);

    if (is_jpeg_file_ext(ext))
      read_image_with<JpegFileReader>(image, filepath.c_str());
    else if (is_png_file_ext(ext))
      read_image_with<PngFileReader>(image, filepath.c_str());
    else if (is_tiff_file_ext(ext))
      read_image_with<TiffFileReader>(image, filepath.c_str());
    else
      throw std::runtime_error{
          format("Image format: %s is either unsupported or invalid",
                 ext.c_str())
              .c_str()};

    auto info = EXIFInfo{};
    if (read_exif_info(info, filepath))
      make_upright_from_exif(image, info.Orientation);
  }

  void imwrite(const Image<Rgb8>& image, const std::string& filepath,
               int quality)
  {
    const auto ext = file_ext(filepath);

    if (is_jpeg_file_ext(ext))
    {
      JpegFileWriter{reinterpret_cast<const unsigned char*>(image.data()),
                     image.width(), image.height(), 3}
          .write(filepath.c_str(), quality);
    }

    else if (is_png_file_ext(ext))
    {
      PngFileWriter{reinterpret_cast<const unsigned char*>(image.data()),
                    image.width(), image.height(), 3}
          .write(filepath.c_str());
    }

    else if (is_tiff_file_ext(ext))
    {
      TiffFileWriter{reinterpret_cast<const unsigned char*>(image.data()),
                     image.width(), image.height(), 3}
          .write(filepath.c_str());
    }
    else
      throw std::runtime_error{"Not a supported or valid image format!"};
  }

} /* namespace Sara */
} /* namespace DO */

// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Graphics.hpp>
#include "ImageIO.hpp"

using namespace DO;
using namespace std;

//! Incorporate these to "DO/Core.hpp"
namespace DO {

  template <typename T, typename U>
  inline void convertColor(Color<T, Rgb>& dst, const Color<U, Rgba>& src)
  {
    convertColor(red(dst), red(src));
    convertColor(green(dst), green(src));
    convertColor(blue(dst), blue(src));
  }

  template <typename T, typename U>
  inline void convertColor(Color<T, Rgba>& dst, const Color<U, Rgb>& src)
  {
    convertColor(red(dst), red(src));
    convertColor(green(dst), green(src));
    convertColor(blue(dst), blue(src));
  }

  template <typename T>
  inline void convertColor(Color<T, Rgb>& dst, const Color<T, Rgba>& src)
  {
    red(dst) = red(src);
    green(dst) = green(src);
    blue(dst) = blue(src);
  }

  template <typename T>
  inline void convertColor(Color<T, Rgba>& dst, const Color<T, Rgb>& src)
  {
    red(dst) = red(src);
    green(dst) = green(src);
    blue(dst) = blue(src);
  }
}

template <typename ImageReader, typename ImageWriter>
void test_image_io(const string& inpath, const string& outpath)
{
  try
  {
    cout << "Try reading file:" << endl << inpath << endl;
    unsigned char *data = 0;
    int w, h, d;
    w = h = d = 0;

    ImageReader readImage(inpath);
    readImage(data, w, h, d);
    if (d == 1) {
      Image<unsigned char> image(&data[0], Vector2i(w,h));
      viewImage(image);
    } else if (d == 3) {
      Image<Rgb8> image(reinterpret_cast<Rgb8 *>(&data[0]), Vector2i(w,h));
      viewImage(image);
    } else if (d == 4) {
      Image<Rgba8> image(reinterpret_cast<Rgba8 *>(&data[0]), Vector2i(w,h));
      viewImage(image);
    }

    cout << "Try writing file:" << endl << outpath << endl;
    ImageWriter writeImage(data, w, h, d);
    writeImage(outpath, 100);

    delete [] data;
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }
}

string fileExtension(const string& filepath)
{
  string ext( filepath.substr(filepath.find_last_of(".") + 1) );
  transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

template <typename T>
bool read(Image<T>& image, const string& filepath)
{
  string ext(fileExtension(filepath));
  bool success = false;
  switch (ext)
  {
  case ".jpeg":
  case ".jpe":
  case ".jfif":
  case ".jfi":
    success = test_image_io<JpegFileReader, JpegFileWriter>(filepath, srcPath("test")+ext);
    break;
  case ".png":
    success = test_image_io<PngFileReader, PngFileWriter>(filepath, srcPath("test")+ext);
    break;
  case ".tif":
  case ".tiff":
    success = test_image_io<TiffFileReader, TiffFileWriter>(filepath, srcPath("test")+ext);
  case default:
    cerr << "Image format: " << ext << " either currently unsupported or invalid" << endl;
  }
  return false;
}

template <typename T, typename ImageReader>
bool readImage(Image<T>& image, const string& filepath)
{
  try
  {
    unsigned char *data = 0;
    int w, h, d;

    ImageReader readImage(filepath);
    readImage(data, w, h, d);
    if (d == 1) {
      Image<unsigned char> tmp(&data[0], Vector2i(w,h), true);
      image = tmp.convert<T>();
    } else if (d == 3) {
      Image<Rgb8> tmp(reinterpret_cast<Rgb8 *>(&data[0]), Vector2i(w,h), true);
      image = tmp.convert<T>();
    } else if (d == 4) {
      Image<Rgba8> tmp(reinterpret_cast<Rgba8 *>(&data[0]), Vector2i(w,h), true);
      image = tmp.convert<T>();
    }
    return true;
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
    return false;
  }
}

int main()
{
  //test_image_io<JpegFileReader, JpegFileWriter>(srcPath("ksmall.jpg"), srcPath("ksmall_write.jpg"));
  //test_image_io<PngFileReader,  PngFileWriter >(srcPath("flower.png"), srcPath("flower_write.png"));
  test_image_io<TiffFileReader,  TiffFileWriter >(srcPath("MARBIBM.TIF"), srcPath("MARBIBM_write.TIF"));

  //Image<Rgb8> image;
  //readImage<Rgb8, TiffFileReader>(image, srcPath("MARBIBM.TIF"));
  return 0;
}
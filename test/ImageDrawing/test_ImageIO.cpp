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

#include <DO/ImageDrawing.hpp>
#include <DO/Graphics.hpp>

using namespace DO;
using namespace std;

namespace DO {

  template <typename T>
  void convertColor(Color<T, Rgb>& dst, const Color<T, Rgba>& src)
  {
    red(dst) = red(src);
    blue(dst) = blue(src);
    green(dst) = green(src);
  }

  template <typename T>
  void convertColor(Color<T, Rgba>& dst, const Color<T, Rgb>& src)
  {
    red(dst) = red(src);
    blue(dst) = blue(src);
    green(dst) = green(src);
  }

  template <typename T, typename U>
  void convertColor(Color<T, Rgb>& dst, const Color<U, Rgba>& src)
  {
    convertColor(red(dst), red(src));
    convertColor(blue(dst), blue(src));
    convertColor(green(dst), green(src));
  }

  template <typename T, typename U>
  void convertColor(Color<T, Rgba>& dst, const Color<U, Rgb>& src)
  {
    convertColor(red(dst), red(src));
    convertColor(blue(dst), blue(src));
    convertColor(green(dst), green(src));
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
  string ext( filepath.substr(filepath.find_last_of(".")) );
  transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

template <typename T>
bool read(Image<T>& image, const string& filepath)
{
  string ext(fileExtension(filepath));
  bool success = true;
  if (ext == ".jpeg" || ext == ".jpe" || ext == ".jfif" || ext == ".jfi")
    test_image_io<JpegFileReader, JpegFileWriter>(filepath, srcPath("test")+ext);
  else if (ext == ".png")
    test_image_io<PngFileReader, PngFileWriter>(filepath, srcPath("test")+ext);
  //else if (ext == ".tif" || ext == ".tiff")
  //  test_image_io<TiffFileReader, TiffFileWriter>(filepath, srcPath("test")+ext);
  else {
    cerr << "Image format: " << ext << " either currently unsupported or invalid" << endl;
    success = false;
  }
  return success;
}

int main()
{
  test_image_io<JpegFileReader, JpegFileWriter>(srcPath("../../datasets/ksmall.jpg"),
                                                srcPath("ksmall_write.jpg"));
  test_image_io<PngFileReader, PngFileWriter>(srcPath("../../datasets/stinkbug.png"),
                                              srcPath("stinkbug_write.png"));
  test_image_io<TiffFileReader, TiffFileWriter>(srcPath("../../datasets/GuardOnBlonde.TIF"),
                                                srcPath("GuardOnBlonde_write.TIF"));
  return 0;
}
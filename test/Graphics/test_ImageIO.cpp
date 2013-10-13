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
#include <cstdio>
#include "ImageIO.hpp"

using namespace DO;
using namespace std;

int main()
{
  string filepath(srcPath("ksmall.jpg"));
  Image<Rgb8> image;
  try
  {
    unsigned char *data = 0;
    int w, h, d;
    w = h = d = 0;

    JpegFileReader reader(filepath);
    reader.read(data, w, h, d);
    Image<Rgb8> image( (Rgb8 *) &data[0], Vector2i(w,h));
    viewImage(image);
    
    delete [] data;
  }
  catch (exception& e)
  {
    cout << e.what() << endl;
  }  

  return 0;
}
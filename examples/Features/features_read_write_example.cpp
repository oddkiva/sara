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

#include <DO/Sara/Features.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageProcessing.hpp>


using namespace std;
using namespace DO::Sara;


const Rgb8& c = Cyan8;


void check_affine_adaptation(const Image<unsigned char>& image,
                             const OERegion& f)
{
  int w = image.width();
  int h = image.height();
  display(image);
  f.draw(Blue8);

  Image<float> flt_image(image.convert<float>());

  float r = 100;
  float patchSz = 2*r;
  Image<float> patch(w, h);
  patch.array().fill(0.f);

  OERegion rg(f);
  rg.center().fill(patchSz/2.f);
  rg.orientation() = 0.f;
  rg.shape_matrix() = Matrix2f::Identity()*4.f / (r*r);

  Matrix3d A(f.affinity().cast<double>());
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patchSz; ++y)
  {
    float v = 2*(y-r)/r;
    for (int x = 0; x < patchSz; ++x)
    {
      float u = 2*(x-r)/r;
      Point3d pp(u, v, 1.);
      pp = A*pp;

      Point2d p;
      p << pp(0), pp(1);

      if (p.x() < 0 || p.x() >= w || p.y() < 0 || p.y() >= h)
        continue;

      patch(x,y) = static_cast<float>(interpolate(flt_image, p));
    }
  }

  Window w1 = active_window();
  Window w2 = create_window(static_cast<int>(patchSz),
                            static_cast<int>(patchSz));
  set_active_window(w2);
  set_antialiasing();
  display(patch);
  rg.draw(Blue8);
  millisleep(1000);
  close_window(w2);

  millisleep(40);
  set_active_window(w1);
}

void read_features(const Image<unsigned char>& image,
                   const string& filepath)
{
  cout << "Reading DoG features... " << endl;
  vector<OERegion> features;
  DescriptorMatrix<float> descriptors;

  cout << "Reading keypoints..." << endl;
  read_keypoints(features, descriptors, filepath);

  for (int i = 0; i < 10; ++i)
    check_affine_adaptation(image, features[i]);

  string ext = filepath.substr(filepath.find_last_of("."), filepath.size());
  string name = filepath.substr(0, filepath.find_last_of("."));
  string copy_filepath = name + "_copy" + ext;
  write_keypoints(features, descriptors, name + "_copy" + ext);

  vector<OERegion> features2;
  DescriptorMatrix<float> descriptors2;
  cout << "Checking written file..." << endl;
  read_keypoints(features2, descriptors2, copy_filepath);

  cout << "Printing the 10 first keypoints..." << endl;
  for(size_t i = 0; i < 10; ++i)
    cout << features[i] << endl;

  // Draw features.
  cout << "Drawing features... ";
  display(image);
  draw_oe_regions(features, Red8);
  cout << "done!" << endl;
  millisleep(1000);
}

GRAPHICS_MAIN()
{
  Image<unsigned char> I;
  load(I, src_path("obama_2.jpg"));

  set_active_window(create_window(I.width(), I.height()));
  set_antialiasing(active_window());
  read_features(I, src_path("test.dogkey"));
  read_features(I, src_path("test.haraffkey"));
  read_features(I, src_path("test.mserkey"));

  return 0;
}
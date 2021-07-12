// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/Features.hpp>
#include <DO/Sara/ImageProcessing.hpp>
#include <DO/Sara/Visualization.hpp>


using namespace std;
using namespace DO::Sara;


const Rgb8& c = Cyan8;


void check_affine_adaptation(const Image<unsigned char>& image,
                             const OERegion& f)
{
  const auto w = image.width();
  const auto h = image.height();
  const auto r = 100.f;
  const auto patch_sz = 2 * r;

  auto gray32f_image = image.convert<float>();
  auto patch = Image<float>{w, h};
  patch.flat_array().fill(0.f);

  display(image);
  draw(f, Blue8);

  auto region = OERegion{f};
  region.center().fill(patch_sz / 2.f);
  region.orientation = 0.f;
  region.shape_matrix = Matrix2f::Identity() * 4.f / (r * r);

  auto A = f.affinity();
  cout << "A=\n" << A << endl;

  for (int y = 0; y < patch_sz; ++y)
  {
    const auto v = float{2 * (y - r) / r};

    for (int x = 0; x < patch_sz; ++x)
    {
      const auto u = float{2 * (x - r) / r};

      const auto P = Point3f{A * Point3f{u, v, 1.}};
      const auto p = Point2d{P.head(2).cast<double>()};

      if (p.x() < 0 || p.x() >= w || p.y() < 0 || p.y() >= h)
        continue;

      patch(x, y) = static_cast<float>(interpolate(gray32f_image, p));
    }
  }

  auto w1 = active_window();
  auto w2 = create_window(static_cast<int>(patch_sz),  //
                          static_cast<int>(patch_sz));
  set_active_window(w2);
  set_antialiasing();
  display(patch);
  draw(region, Blue8);
  millisleep(1000);
  close_window(w2);

  millisleep(40);
  set_active_window(w1);
}

void read_features(const Image<unsigned char>& image, const string& filepath)
{
  cout << "Reading DoG features... " << endl;
  auto features = vector<OERegion>{};
  Tensor_<float, 2> descriptors;

  cout << "Reading keypoints..." << endl;
  read_keypoints(features, descriptors, filepath);

  for (auto i = 0; i < 10; ++i)
    check_affine_adaptation(image, features[i]);

  auto ext = filepath.substr(filepath.find_last_of("."), filepath.size());
  auto name = filepath.substr(0, filepath.find_last_of("."));
  auto copy_filepath = name + "_copy" + ext;
  write_keypoints(features, descriptors, name + "_copy" + ext);

  auto features2 = vector<OERegion>{};
  Tensor_<float, 2> descriptors2;
  cout << "Checking written file..." << endl;
  read_keypoints(features2, descriptors2, copy_filepath);

  cout << "Printing the 10 first keypoints..." << endl;
  for (auto i = 0; i < 10; ++i)
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
  auto I = Image<unsigned char>{};
  load(I, src_path("obama_2.jpg"));

  set_active_window(create_window(I.width(), I.height()));
  set_antialiasing(active_window());
  read_features(I, src_path("test.dogkey"));
  read_features(I, src_path("test.haraffkey"));
  read_features(I, src_path("test.mserkey"));

  return 0;
}

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

#include <DO/Sara/Core/Timer.hpp>
#include <DO/Sara/ImageDrawing.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <gtest/gtest.h>

using namespace DO;
using namespace std;

TEST(DO_Sara_ImageDrawing_Test, imageFileReadingTest)
{
  string filePaths[] = {
    srcPath("../../datasets/ksmall.jpg"),
    srcPath("../../datasets/stinkbug.png"),
    srcPath("../../datasets/All.tif")
  };

  HighResTimer timer;
  double elapsed1, elapsed2;

  for (int i = 0; i < 3; ++i)
  {
    Image<Rgb8> image;

    timer.restart();
    ASSERT_TRUE(imread(image, filePaths[i]));
    ASSERT_NE(image.sizes(), Vector2i::Zero());
    elapsed1 = timer.elapsedMs();
    cout << "ImageIO loading time = " << elapsed1 << " ms" << endl;
    viewImage(image);

    timer.restart();
    ASSERT_TRUE(load(image, filePaths[i]));
    ASSERT_NE(image.sizes(), Vector2i::Zero());
    elapsed2 = timer.elapsedMs();
    cout << "Qt-based loading time = " << elapsed2 << " ms" << endl;
    viewImage(image);

    cout << "Speed factor = " << elapsed2/elapsed1 << endl << endl;

    EXIFInfo exifInfo;
    if (readExifInfo(exifInfo, filePaths[i]))
      print(exifInfo);
  }
}

TEST(DO_Sara_ImageDrawing_Test, imageExifOriTest)
{
  vector<string> filePaths;
  getImageFilePaths(filePaths, "C:/data/David-Ok-Iphone4S");

  HighResTimer timer;
  double elapsed;

  bool viewImageCollection = true;
  if (viewImageCollection)
  openGraphicsView(1024, 768);

  for (size_t i = 0; i < filePaths.size(); ++i)
  {
    const string& filePath = filePaths[i];
    Image<Rgb8> image;
    EXIFInfo exifInfo;

    timer.restart();
    ASSERT_TRUE(imread(image, filePath));
    elapsed = timer.elapsedMs();
    cout << "Load time = " << elapsed << " ms" << endl;

    ASSERT_NE(image.sizes(), Vector2i::Zero());
    ASSERT_TRUE(readExifInfo(exifInfo, filePath));

    if (viewImageCollection)
      addImage(image, true);

    print(exifInfo);
  }

  while (getKey() != KEY_ESCAPE && viewImageCollection);
    closeWindow();
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
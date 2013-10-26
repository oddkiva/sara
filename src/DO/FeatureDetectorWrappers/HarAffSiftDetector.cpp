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

#include <DO/FeatureDetectorWrappers.hpp>
#include <DO/Graphics.hpp>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <string>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace DO
{
  void HarAffSiftDetector::run(std::vector<OERegion>& features,
                               DescriptorMatrix<float>& descriptors,
                               const Image<uchar>& I,
                               bool specifyThres,
                               double HarrisT) const
  {
    using namespace std;

    // Save image in PNG format (required by Mikolajczyk's software.)
    int tn = 0;
#ifdef _OPENMP
    tn = omp_get_thread_num();
#endif
    const string fileName("TMP_IMAGE_"+toString(tn)+".png");
    if ( !save(I, fileName, 100) )
      cerr << "Error: cannot save image!" << std::endl;

    // Run Mikolajczyk's binary.
    // I need to rerun twice the binary just to get the key orientation.
    // This is stupid but I don't know other alternative.
    // Also, Mikolajczyk's file format #2 is strange.
    //
    // I also discovered the second binary version of Mikolajczyk's is buggy 
    // with the Harris-affine features (from featurespace.org) when I run 
    // it with OpenMP.
    cout << externBinPath("MikolajczykFeatureExtractor.exe ") << endl;
    const string program(externBinPath("MikolajczykFeatureExtractor.exe "));
    string thres;
    if (specifyThres)
    {
      ostringstream oss;
      oss << HarrisT;
      thres = "-harThres " + oss.str() + " ";
    }
    const string command(  program + "-haraff -sift " + thres
               + "-i " + fileName 
               + " -o1 TMP_DESC1_" + toString(tn));
    
    const string command2(  program + "-haraff -sift " + thres
                + "-i " + fileName 
                + " -o2 TMP_DESC2_" + toString(tn));
    std::system(command.c_str());
    std::system(command2.c_str());

    // Parse the descriptor files.
    ifstream descFile1(string("TMP_DESC1_"+ toString(tn)).c_str());
    if (!descFile1.is_open())
      cerr << "Cant open file " << "TMP_DESC1_" << toString(tn) << endl;
    ifstream descFile2(string("TMP_DESC2_"+ toString(tn)).c_str());
    if(!descFile2.is_open())
      cerr << "Cant open file " << "TMP_DESC2_" << toString(tn) << endl;

    int descDim, sz;
    descFile1 >> descDim >> sz;
    // Ignore the 2 first lines in the second file.
    string line;
    for (int i = 0; i < 2; ++i)
      getline(descFile2, line);

    // Check the dimension of the descriptor
    if (descDim != 128)
    {
      cerr << "Descriptor dimension does not match with SIFT descriptor"
         << " dimension!" << endl;
      exit(-1);
    }

    // Get the array of features.
    features.resize(sz);
    descriptors.resize(sz, 128);
    for (int i = 0; i < sz; ++i)
    {
      OERegion& f = features[i];
      // Feature type
      f.type() = OERegion::HarAff;
      // Center
      descFile1 >> f.center();
      // Shape matrix
      descFile1 >> f.shapeMat()(0,0) 
                >> f.shapeMat()(1,0)
                >> f.shapeMat()(1,1);
      f.shapeMat()(0,1) = f.shapeMat()(1,0);
      // SIFT descriptor
      descFile1 >> descriptors[i];


      // Ignore the following variables before getting the orientation
      // from descFile2.
      float cornerness, objIndex, ptType, laplacianVal, extremumType, patchSize;
      Point2f p;
      Matrix2f m;
      descFile2 >> p; 
      descFile2 >> cornerness;
      descFile2 >> patchSize >> f.orientation();

      // Ignore the following variables.
      descFile2 >> objIndex >> ptType >> laplacianVal >> extremumType;
      descFile2 >> m;
      Matrix<double, 128, 1> desc;
      descFile2 >> desc;
    }
  }

} /* namespace DO */
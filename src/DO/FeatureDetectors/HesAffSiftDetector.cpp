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

#include <DO/FeatureDetectors.hpp>
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

  std::vector<Keypoint> HesAffSiftDetector::run(const Image<uchar>& I,
							                                  bool specifyThres,
							                                  double HessianT) const
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
		const string program(externBinPath("MikolajczykFeatureExtractor2.exe "));
		string thres;
		if(specifyThres)
		{
			ostringstream oss;
			oss << HessianT;
			thres = "-hesThres " + oss.str() + " ";
		}
		const string command(  program + "-hesaff -sift " + thres
							 + "-i " + fileName 
							 + " -o1 TMP_DESC1_" + toString(tn) );

		const string command2(  program + "-hesaff -sift " + thres
							  + "-i " + fileName 
							  + " -o2 TMP_DESC2_" + toString(tn) );

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
		const int siftDim = 128;
		descFile1 >> descDim >> sz;

		// Ignore the 4 first lines in the second file.
		string line;
		for(int i = 0; i < 4; ++i)
			getline(descFile2, line);

		assert(descDim == siftDim);
		if(descDim != siftDim)
		{
			cerr << "Descriptor dimension does not match with SIFT descriptor"
				 << " dimension!" << endl;
			exit(-1);
		}

		// Get the array of features
		vector<Keypoint>  features;
		features.resize(sz);
		for (int i = 0; i < sz; ++i)
		{
      Keypoint& k = features[i];
      OERegion& f = k.feat();
      Desc128f& d = k.desc();
			// Feature type
			f.type() = OERegion::HesAff;
			// Position
			descFile1 >> f.center();
			// Ellipse parameters
			// Shape matrix
      descFile1 >> f.shapeMat()(0,0) 
        >> f.shapeMat()(1,0)
        >> f.shapeMat()(1,1);
      f.shapeMat()(0,1) = f.shapeMat()(1,0);
			// SIFT descriptor
      descFile1 >> d;
			
			// Ignore the following variables before getting the orientation
			// from the second file.
			float psize, dDim, cornerness, objIndex, ptType, laplacianVal, 
        extremumType, scale;
			Point2f p;
			Matrix2f m;
			descFile2 >> psize >> dDim >> p >> cornerness >> scale;
			
			// Now get the orientation
			descFile2 >> f.orientation();

			// Ignore the following variables.
			descFile2 >> objIndex >> ptType >> laplacianVal >> extremumType >> m;
			// Mikolajczyk's strange array...
			Matrix<double, 23, 1> strangeArray;
			descFile2 >> strangeArray;
			Matrix<double, 128, 1> desc1;
			descFile2 >> desc1;
		}

		return features;
	}

} /* namespace DO */
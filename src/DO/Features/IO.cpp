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

#include <DO/Features.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

namespace DO {

  inline float make_round(float x, int n)
  {
    float p=pow(10.f,n);
    x*=(p);
    x=(x>=0)?floor(x):ceil(x);
    return x/p;      
  }

  inline float toDegree(float radian)
  {
    return radian/float(M_PI)*180.f;
  }

  std::ostream& operator<<(std::ostream& os, const Keypoint& k)
  {
    os << "Feature type:\t";
    switch (k.feat().type())
    {
    case PointFeature::DoG:
      os << "DoG" << std::endl;
      break;
    case PointFeature::HarAff:
      os << "HarrisAffine" << std::endl;
      break;
    case PointFeature::HesAff:
      os << "HessianAffine" << std::endl;
      break;
    case PointFeature::MSER:
      os << "MSER" << std::endl;
      break;
    default:
      break;
    }
    os << "position:\n" << k.feat().coords() << std::endl;;
    os << "shape matrix:\n" << k.feat().shapeMat() << std::endl;
    os << "orientation:\t" << toDegree(k.feat().orientation()) << " degrees"
       << std::endl;

    return os;
  }

  bool readKeypoints(std::vector<Keypoint>& keys, const std::string& name,
                     bool bundlerFormat)
  {
    std::ifstream f(name.c_str());
    if (!f.is_open()) {
      std::cerr << "Cant open file " << name << std::endl;    
      return false;
    }
    int nb,sz;
    f >> nb >> sz;
    if (sz != 128) {
      std::cerr << "Bad desc size" << std::endl;    
      return false;
    }
    keys.resize(nb);
    for (int i = 0; i < nb; ++i)
    {
      OERegion& feat = keys[i].feat();
      Desc128f& desc = keys[i].desc();

      if (bundlerFormat)
      {
        float scale;
        f >> feat.y() >> feat.x();
        f >> scale;
        feat.shapeMat() = Matrix2f::Identity() / (scale*scale);
        f >> feat.orientation();
      }
      else
      {
        f >> feat.coords() >> feat.shapeMat() >> feat.orientation();
        double dFT;
        f >> dFT;
        feat.type() =  OERegion::Type(int(dFT));
      }

      f >> desc;
    }
    f.close();
    return true;
  }

  bool writeKeypoints(const std::vector<Keypoint>& keys, const std::string& name, 
                      bool writeForBundler)
  {
    std::ofstream f(name.c_str());
    if (!f.is_open()) {
      std::cerr << "Cant open file" << std::endl;    
      return false;
    }

    f << keys.size() << " " << 128 << std::endl;
    for(size_t i = 0; i < keys.size(); ++i)
    {
      const OERegion& feat = keys[i].feat();
      const Desc128f& desc = keys[i].desc();

      if (writeForBundler)
      {
        f << make_round(feat.y(), 2) << ' ';
        f << make_round(feat.x(), 2) << ' ';
        f << make_round(feat.scale(), 2) << ' ';
        f << make_round(feat.orientation(), 3) << std::endl;
      }
      else
      {
        f << feat.x() << ' ' << feat.y() << std::endl;
        f << feat.shapeMat().array() << std::endl;
        f << feat.orientation() << std::endl;
        f << double(feat.type()) << std::endl;
        f << desc << std::endl;
      }
    }
    f.close();
    return true;
  }
}
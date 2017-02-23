// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#include "Stat.hpp"

using namespace std;

namespace DO {

  ostream& operator<<(ostream& os, const Stat& s)
  {
    os << "size   = " << s.size   << endl;
    os << "min    = " << s.min    << endl;
    os << "max    = " << s.max    << endl;
    os << "mean   = " << s.mean   << endl;
    os << "median = " << s.median << endl;
    os << "sigma  = " << s.sigma  << endl;
    return os;
  }

  void writeStats(ofstream& out, const vector<Stat>& stats)
  {
    out << "size\t";
    for (size_t i = 0; i != stats.size(); ++i)
      out << stats[i].size << "\t";
    out << endl;
    out << "min\t";
    for (size_t i = 0; i != stats.size(); ++i)
      out << stats[i].min << "\t";
    out << endl;
    out << "max\t";
    for (size_t i = 0; i != stats.size(); ++i)
      out << stats[i].max << "\t";
    out << endl;
    out << "mean\t";
    for (size_t i = 0; i != stats.size(); ++i)
      out << stats[i].mean << "\t";
    out << endl;
    out << "median\t";
    for (size_t i = 0; i != stats.size(); ++i)
      out << stats[i].median << "\t";
    out << endl;
    out << "sigma\t";
    for (size_t i = 0; i != stats.size(); ++i)
      out << stats[i].sigma << "\t";
    out << endl;
  }

} /* namespace DO */
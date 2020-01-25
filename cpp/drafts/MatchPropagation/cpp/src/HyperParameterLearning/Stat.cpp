// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file
/*!
 *  This file implements a part of the method published in:
 *
 *  Efficient and Scalable 4th-order Match Propagation
 *  David Ok, Renaud Marlet, and Jean-Yves Audibert.
 *  ACCV 2012, Daejeon, South Korea.
 */

#include "Stat.hpp"


using namespace std;


namespace DO::Sara {

  ostream& operator<<(ostream& os, const Stat& s)
  {
    os << "size   = " << s.size << endl;
    os << "min    = " << s.min << endl;
    os << "max    = " << s.max << endl;
    os << "mean   = " << s.mean << endl;
    os << "median = " << s.median << endl;
    os << "sigma  = " << s.sigma << endl;
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

} /* namespace DO::Sara */

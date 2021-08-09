// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <fstream>
#include <string>
#include <sstream>

#include <DO/Sara/Match.hpp>


using namespace std;


namespace DO { namespace Sara {

  ostream & operator<<(ostream & os, const Match& m)
  {
    os << "source=" << m.x_pos().transpose()
       << " target=" << m.y_pos().transpose() << endl;
    os << "score=" << m.score() << " " << "rank=" << m.rank();
    return os;
  }

  bool write_matches(const vector<Match>& matches, const string& filepath)
  {
    ofstream file(filepath.c_str());
    if (!file.is_open()) {
      cerr << "Cant open file" << filepath << endl;
      return false;
    }

    file << matches.size() << endl;
    for(auto m = matches.begin(); m != matches.end(); ++m)
      file << m->x_index() << ' ' << m->y_index() << ' '
           << m->rank() << ' ' << m->score() << endl;

    return true;
  }

  bool read_matches(vector<Match>& matches,
                    const string& filepath,
                    float score_thres)
  {
    if (!matches.empty())
      matches.clear();

    ifstream file(filepath.c_str());
    if (!file.is_open())
    {
      cerr << "Cant open file: " << filepath << endl;
      return false;
    }

    size_t match_count;
    file >> match_count;

    matches.reserve(match_count);
    for (size_t i = 0; i < match_count; ++i)
    {
      Match m;

      file >> m.x_index() >> m.y_index() >> m.rank() >> m.score();
      if(m.score() > score_thres)
        break;

      matches.push_back(m);
    }

    return true;
  }

  bool read_matches(vector<Match>& matches,
                    const vector<OERegion>& source_keys,
                    const vector<OERegion>& target_keys,
                    const string& fileName,
                    float score_thres)
  {
    if (!matches.empty())
      matches.clear();

    std::ifstream file(fileName.c_str());
    if (!file.is_open()) {
      std::cerr << "Cant open file: " << fileName << std::endl;
      return false;
    }

    std::size_t match_count;
    file >> match_count;

    matches.reserve(match_count);
    for (size_t i = 0; i < match_count; ++i)
    {
      Match m;

      file >> m.x_index() >> m.y_index() >> m.rank() >> m.score();
      m.x_pointer() = &source_keys[m.x_index()];
      m.y_pointer() = &target_keys[m.y_index()];

      if(m.score() > score_thres)
        break;

      matches.push_back(m);
    }

    return true;
  }

} /* namespace Sara */
} /* namespace DO */

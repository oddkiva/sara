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

#include <DO/Sara/Graphics.hpp>

#include "GrowMultipleRegions.hpp"


using namespace std;


namespace DO::Sara {

  GrowMultipleRegions::GrowMultipleRegions(const vector<Match>& M,
                                           const GrowthParams& params,
                                           int verbose)
    : G_(M, params.K(), float(params.rho_min()))
    , params_(params)
    , verbose_(verbose)
  {
  }

  vector<Region> GrowMultipleRegions::
  operator()(size_t N, RegionGrowingAnalyzer* pAnalyzer,
             const PairWiseDrawer* pDrawer)
  {
    vector<Region> R_;
    R_.reserve(1000);

    Region allR;
    size_t maxRegionSize = numeric_limits<size_t>::max();

    size_t n = 0;
    for (size_t m = 0; m != G_.size(); ++m)
    {
      // Can we use the following match as a seed?
      if (allR.find(m))
        continue;

      // Attempt region growing.
      GrowRegion growRegion(m, G_, params_);
      pair<Region, vector<size_t>> result =
          growRegion(R_, maxRegionSize, pDrawer, pAnalyzer);

      // Merge regions if overlapping!
      if (!result.second.empty())
      {
        if (pAnalyzer)
          pAnalyzer->increment_num_fusions();

        // Try growing $T$ different regions at most.
        if (verbose_ >= 2)
        {
          cout << "[" << n << "] with match M[" << m << "]: Overlapping"
               << endl;
          cout << "region size = " << result.first.size() << endl;
        }
        // if (pDrawer)
        //{
        //  for (size_t i = 0; i != result.second.size(); ++i)
        //    cout << result.second[i] << endl;
        //  pDrawer->displayImages();
        //  checkRegions(R_, pDrawer);
        //  result.first.view(G_.M(), *pDrawer, Cyan8);
        //  getKey();
        //}

        if (verbose_ >= 2)
          cout << "Before merging: number of regions = " << R_.size() << endl;
        // Merge the overlapping regions.
        merge_regions(R_, result);
        // Remember not to grow from the following matches.
        mark_reliable_matches(allR, result.first);

        // Check visually the set of regions.
        if (verbose_ >= 2)
          cout << "After merging: number of regions = " << R_.size() << endl;
        // check_regions(R_, pDrawer);
      }
      // if the region has significant size, add it to the list of region.
      else if (result.first.size() > 7)
      {
        // Try growing $T$ different regions at most.
        if (verbose_ >= 2)
        {
          cout << "[" << n << "] with match M[" << m
               << "]: Regular region growing" << endl;
          cout << "region size = " << result.first.size() << endl;
        }
        // Add to the partition of consistent regions.
        R_.push_back(result.first);
        // Remember not to grow from the following matches.
        mark_reliable_matches(allR, result.first);
        // Check visually the set of regions.
        // if (pDrawer)
        //{
        //  checkRegions(R_, pDrawer);
        //  milliSleep(250);
        //}
      }

      ++n;
      if (n == N)
        break;
    }

    if (verbose_ >= 1)
    {
      cout << "FINISHED GROWING MULTIPLE REGIONS:" << endl;
      cout << "number of regions = " << R_.size() << endl;
      cout << "number of matches = " << allR.size() << endl;
    }
    if (pDrawer)
    {
      check_regions(R_, pDrawer);
      // get_key();
    }
    if (pAnalyzer)
    {
      pAnalyzer->set_num_regions(int(R_.size()));
      pAnalyzer->set_num_attempted_growths(int(N));
    }
    return R_;
  }

  void GrowMultipleRegions::mark_reliable_matches(Region& allR,
                                                  const Region& R) const
  {
    // Remember not to grow from the following matches.
    for (Region::iterator m = R.begin(); m != R.end(); ++m)
      allR.insert(*m);
  }

  void GrowMultipleRegions::merge_regions(
      std::vector<Region>& Rs, const pair<Region, vector<size_t>>& result) const
  {
    // Merge regions.
    if (verbose_ >= 2)
      cout << "Merging regions:" << endl;
    Region mergedR;
    for (size_t i = 0; i != result.second.size(); ++i)
    {
      size_t index = result.second[i];
      if (verbose_ >= 2)
        cout << index << " ";
      for (Region::iterator m = Rs[index].begin(); m != Rs[index].end(); ++m)
        mergedR.insert(*m);
      for (Region::iterator m = result.first.begin(); m != result.first.end();
           ++m)
        mergedR.insert(*m);
    }
    if (verbose_ >= 2)
      cout << endl;

    // Erase the overlapping regions and put the region resulting from the
    // merging
    vector<Region> newRs;
    newRs.reserve(1000);
    // Find the regions we have to keep.
    vector<int> indicesToKeep(Rs.size(), 1);
    for (size_t i = 0; i != result.second.size(); ++i)
      indicesToKeep[result.second[i]] = 0;
    // Store the desired regions.
    newRs.push_back(mergedR);
    for (size_t i = 0; i != Rs.size(); ++i)
      if (indicesToKeep[i] == 1)
        newRs.push_back(Rs[i]);
    Rs.swap(newRs);
  }

  void GrowMultipleRegions::check_regions(const vector<Region>& RR,
                                          const PairWiseDrawer* pDrawer) const
  {
    if (pDrawer)
    {
      pDrawer->display_images();
      srand(500);
      for (size_t i = 0; i != RR.size(); ++i)
      {
        Rgb8 c(rand() % 256, rand() % 256, rand() % 256);
        RR[i].view(G_.M(), *pDrawer, c);
      }
    }
  }

}  // namespace DO::Sara

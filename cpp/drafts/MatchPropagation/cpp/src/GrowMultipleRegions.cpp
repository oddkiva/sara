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

#include <DO/Sara/Core/DebugUtilities.hpp>
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
  operator()(size_t N, RegionGrowingAnalyzer* analyzer,
             const PairWiseDrawer* drawer)
  {
    vector<Region> R_;
    R_.reserve(1000);

    Region all_R;
    size_t max_region_size = numeric_limits<size_t>::max();

    size_t n = 0;
    for (size_t m = 0; m != G_.size(); ++m)
    {
      // Can we use the following match as a seed?
      if (all_R.find(m))
        continue;

      // Attempt region growing.
      GrowRegion growRegion(m, G_, params_);
      if (verbose_ >= 3)
        growRegion.set_verbose(true);

      pair<Region, vector<size_t>> result =
          growRegion(R_, max_region_size, drawer, analyzer);

      // Merge regions if overlapping!
      if (!result.second.empty())
      {
        if (analyzer)
          analyzer->increment_num_fusions();

        // Try growing $T$ different regions at most.
        if (verbose_ >= 2)
        {
          SARA_DEBUG << "[" << n << "] with match M[" << m << "]: Overlapping"
                     << endl;
          SARA_DEBUG << "region size = " << result.first.size() << endl;
        }

        //if (drawer)
        //{
        //  for (size_t i = 0; i != result.second.size(); ++i)
        //    SARA_DEBUG << result.second[i] << endl;
        //  drawer->display_images();
        //  check_regions(R_, drawer);
        //  result.first.view(G_.M(), *drawer, Cyan8);
        //  get_key();
        //}

        if (verbose_ >= 2)
          SARA_DEBUG << "Before merging: number of regions = " << R_.size()
                     << endl;
        // Merge the overlapping regions.
        merge_regions(R_, result);
        // Remember not to grow from the following matches.
        mark_reliable_matches(all_R, result.first);

        // Check visually the set of regions.
        if (verbose_ >= 2)
          SARA_DEBUG << "After merging: number of regions = " << R_.size()
                     << endl;
      }
      // if the region has significant size, add it to the list of region.
      else if (result.first.size() > 7)
      {
        // Try growing $T$ different regions at most.
        if (verbose_ >= 2)
        {
          SARA_DEBUG << "[" << n << "] with match M[" << m
                     << "]: Regular region growing" << endl;
          SARA_DEBUG << "region size = " << result.first.size() << endl;
        }
        // Add to the partition of consistent regions.
        R_.push_back(result.first);
        // Remember not to grow from the following matches.
        mark_reliable_matches(all_R, result.first);
      }

      ++n;
      if (n == N)
        break;
    }

    if (verbose_ >= 1)
    {
      SARA_DEBUG << "FINISHED GROWING MULTIPLE REGIONS:" << endl;
      SARA_DEBUG << "number of regions = " << R_.size() << endl;
      SARA_DEBUG << "number of matches = " << all_R.size() << endl;
    }

    if (drawer)
      check_regions(R_, drawer);

    if (analyzer)
    {
      analyzer->set_num_regions(int(R_.size()));
      analyzer->set_num_attempted_growths(int(N));
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
      SARA_DEBUG << "Merging regions:" << endl;
    Region merged_R;
    for (size_t i = 0; i != result.second.size(); ++i)
    {
      size_t index = result.second[i];
      if (verbose_ >= 2)
        SARA_DEBUG << index << " ";
      for (Region::iterator m = Rs[index].begin(); m != Rs[index].end(); ++m)
        merged_R.insert(*m);
      for (Region::iterator m = result.first.begin(); m != result.first.end();
           ++m)
        merged_R.insert(*m);
    }
    if (verbose_ >= 2)
      SARA_DEBUG << endl;

    // Erase the overlapping regions and put the region resulting from the
    // merging.
    vector<Region> new_Rs;
    new_Rs.reserve(1000);
    // Find the regions we have to keep.
    vector<int> indices_to_keep(Rs.size(), 1);
    for (size_t i = 0; i != result.second.size(); ++i)
      indices_to_keep[result.second[i]] = 0;
    // Store the desired regions.
    new_Rs.push_back(merged_R);
    for (size_t i = 0; i != Rs.size(); ++i)
      if (indices_to_keep[i] == 1)
        new_Rs.push_back(Rs[i]);
    Rs.swap(new_Rs);
  }

  void GrowMultipleRegions::check_regions(const vector<Region>& RR,
                                          const PairWiseDrawer* drawer) const
  {
    if (drawer)
    {
      drawer->display_images();
      srand(500);
      for (size_t i = 0; i != RR.size(); ++i)
      {
        Rgb8 c(rand() % 256, rand() % 256, rand() % 256);
        RR[i].view(G_.M(), *drawer, c);
      }
    }
  }

}  // namespace DO::Sara

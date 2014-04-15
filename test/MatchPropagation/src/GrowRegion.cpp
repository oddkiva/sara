// ========================================================================== //
// This file is part of DO++ MatchPropagation which was presented in:
//
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ===========================================================================

#include "GrowRegion.hpp"
#include <DO/Graphics.hpp>
#include "LocalAffineConsistency.hpp"

using namespace std;

//#define DEBUG
#include "GrowRegionDebugMacros.hpp"

#ifdef _OPENMP
# include <omp.h>
#endif

namespace DO {

  // ======================================================================== //
  // Constructor
  GrowRegion::GrowRegion(size_t seed,
                         DynamicMatchGraph& g,
                         const GrowthParams& params,
                         bool verbose)
    : seed_(seed), G_(g), params_(params), verbose_(verbose)
  {
  }

  // ======================================================================== //
  // The main member functions.
  Region GrowRegion::operator()(size_t maxRegionSize,
                                const PairWiseDrawer *pDrawer,
                                RegionGrowingAnalyzer *pAnalyzer)
  {
    // Dummy variable.
    vector<Region> RR;
    return operator()(RR, maxRegionSize, pDrawer, pAnalyzer).first;
  }

  pair<Region, vector<size_t> >
  GrowRegion::operator()(const vector<Region>& RR,
                         size_t maxRegionSize,
                         const PairWiseDrawer *pDrawer,
                         RegionGrowingAnalyzer *pAnalyzer)
  {
    Region R;
    RegionBoundary dR(M());
    vector<size_t> indices;
    // Try initializing the region with an affine-consistent quadruple.
    if (!initAffQuad(R, dR, pDrawer, pAnalyzer))
      return make_pair(R, indices);

    // If initialization is successful, grow the region regularly.
    grow(RR, R, dR, indices, maxRegionSize, pDrawer, pAnalyzer);
    return make_pair(R, indices);
  }

  // ======================================================================== //
  // 1. Try initializing the region with an affine-consistent quadruple.
  bool GrowRegion::initAffQuad(Region& R, RegionBoundary& dR,
                               const PairWiseDrawer *pDrawer,
                               RegionGrowingAnalyzer *pAnalyzer)
  {
    // Initialize the region $R$ and the region boundary $\partial R$.
    updateRegionAndBoundary(R, dR, seed_);
    CHECK_STATE_AFTER_ADDING_SEED_MATCH;

    // Choose a seed triple 't' heuristically.
    // 'q = { t, m };'
    size_t q[4];
    // 'q[i] = t[i]' for $i \in \{ 0, 1, 2 \}$.
    // 'q[3] == m'.    
    if (!buildSeedTriple(q, dR))
       return false;

    for (int i = 0; i < 3; ++i)
    {
      updateRegionAndBoundary(R, dR, q[i]);
      CHECK_INCREMENTAL_SEED_TRIPLE_CONSTRUCTION;
    }
    // Find a match $m$ such that the quadruple $q = (t, m)$ is affine-consistent.
    if (verbose_)
      printStage("Looking for an affine-consistent seed quadruple");
    bool foundAffSeedQuad = false;
    for (RegionBoundary::iterator m = dR.begin(); m != dR.end(); ++m)
    {
      q[3] = m.index();
      CHECK_CANDIDATE_FOURTH_MATCH_FOR_SEED_QUADRUPLE;
      int very_spurious = false;
      foundAffSeedQuad = affineConsistent(q, very_spurious, pDrawer);
      if (foundAffSeedQuad)
      {
        updateRegionAndBoundary(R, dR, q[3]);
        CHECK_GROWING_STATE_AFTER_FINDING_AFFINE_SEED_QUADRUPLE;
        break;
      }
    }
    return foundAffSeedQuad;
  }

  // 2. If initialization is successful, grow the region regularly.
  //    However, the growing process stops if it intersects with other 
  //    regions.
  void GrowRegion::grow(const std::vector<Region>& RR,
                        Region& R, RegionBoundary& dR,
                        vector<size_t>& indices,
                        size_t maxAllowedSize,
                        const PairWiseDrawer *pDrawer,
                        RegionGrowingAnalyzer *pAnalyzer)
  {
    if (verbose_)
      printStage("Growing region regularly");

//#define PROFILE_REGION_GROWING
#ifdef PROFILE_REGION_GROWING
    Timer t;
    double elapsed;
#endif

    bool intersection = false;
    while (!dR.empty())
    {
      // If the region has some critical size, stop growing.
      // This may be useful for object matching.
      if (R.size() > maxAllowedSize)
        break;

      // ==================================================================== //
      // Find all affine-consistent quadruples within the region boundary.
      // This process is computationally intensive so we are experimenting with
      // parallelization.
      //
      // 1. Copy everything in an array.
      vector<size_t> vec_dR;
      vec_dR.reserve(dR.size());
      for (RegionBoundary::iterator m = dR.begin(); m != dR.end(); ++m)
        vec_dR.push_back(m.index());
      
      // ==================================================================== //
      // 2. Construct an array of boolean 'vector<int> good'.
      //    where '0 == false' and '1 == true'.
      //    Every match $m \in \partial R$ are assumed to be no good.
      vector<int> isGoodMatch(vec_dR.size(), 0);
      
      // ==================================================================== //
      // 3. Update and store all the neighborhoods $\mathcal{N}_K(m)$ for 
      //    $m \in \partial R$.
      //    THIS IS LIKELY NOT PARALLELIZABLE BUT THIS IS THE BOTTLENECK.
#ifdef PROFILE_REGION_GROWING
      // =============== TIC (NEIGHBORHOOD COMPUTATION TIME) ================ //
      t.restart();
      // ==================================================================== //
#endif
      vector<vector<size_t> > N_K_cap_R(vec_dR.size());
      G_.updateN_K(vec_dR);
#ifdef PROFILE_REGION_GROWING
      // =============== TOC (NEIGHBORHOOD COMPUTATION TIME) ================ //
      elapsed = t.elapsed();
      cout << "N_K_cap_R.size() = " << N_K_cap_R.size() << endl;
      cout << "Neighborhood computation time = " << elapsed << " s" << endl;
#endif
      // ==================================================================== //
      for (size_t m = 0; m != vec_dR.size(); ++m)
        N_K_cap_R[m] = get_N_K_m_cap_R(vec_dR[m], R);


      // ==================================================================== //
      // 4. Test if $m \in \partial R$ is affine-consistent.
#ifdef PROFILE_REGION_GROWING
      // =============== TIC (AFFINE CONSISTENCY CHECK TIME) ================ //
      t.restart();
      // ==================================================================== //
#endif
      vector<vector<size_t> > q(vec_dR.size());     // Set of quadruples.
#pragma omp parallel for
      for (int m = 0; m < vec_dR.size(); ++m)
        q[m].resize(4);
      vector<int> spurious(vec_dR.size(), 0);       // $m$ is spurious.
#pragma omp parallel for
      for (int m = 0; m < vec_dR.size(); ++m)
      {
        q[m][3] = vec_dR[m];
        CHECK_CANDIDATE_MATCH_AND_GROWING_STATE;

        if (!findTriple(&q[m][0], N_K_cap_R[m], pDrawer))
          continue;

        if (affineConsistent(&q[m][0], spurious[m], pDrawer))
        {
          isGoodMatch[m] = 1;
#pragma omp critical
          {
            analyzeQuadruple(&q[m][0], pAnalyzer);
          }
        }
      }
#ifdef PROFILE_REGION_GROWING
      // =============== TOC (AFFINE CONSISTENCY CHECK TIME) ================ //
      elapsed = t.elapsed();
      cout << "Affine consistency check time = " << elapsed << " s" << endl;
      // ==================================================================== //
#endif
      // ==================================================================== //
      // 5. Count the number of good matches.
      size_t num_good_matches = 0;
      for (size_t m = 0; m != vec_dR.size(); ++m)
        if (isGoodMatch[m] == 1)
          ++num_good_matches;

      // ==================================================================== //
      // 6. Check if there is an intersection, i.e. $\#\{q \cap R\} \geq 3 ?$
      if (num_good_matches != 0)
      {
        intersection = false;
        for (size_t m = 0; m != vec_dR.size(); ++m)
        {
          if (overlap(indices, RR, &q[m][0]) && isGoodMatch[m] == 1)
          {
            intersection = true;
            break;
          }
        }
      }

      // ==================================================================== //
      // 7. Update the growing region with the found consistent matches.
      for (size_t m = 0; m != vec_dR.size(); ++m)
      {
        if (isGoodMatch[m] == 0)
          continue;
        updateRegionAndBoundary(R, dR, vec_dR[m]);
        if (pDrawer)
        {
          //checkGrowingState(R, dR, pDrawer, false);
          pDrawer->drawMatch(M(vec_dR[m]), Green8);
        }
      }

      // ==================================================================== //
      // 8. Do we go to the next iteration in the region growing process?
      if (num_good_matches != 0 && intersection)
      {
        if (verbose_)
        {
          cout << "\n\n[INTERRUPTING REGION GROWING:] ";
          cout << "$R$ intersects with the following regions:" << endl;
          for (size_t i = 0; i != indices.size(); ++i)
            cout << indices[i] << " ";
          cout << endl;
          cout << "Final region size = " << R.size() << endl;
          //getKey();
        }
        break;
      }
      if (num_good_matches == 0)
      {
        if (verbose_)
        {
          cout << "\n\n[Finished region growing!] Check indices.size() = " 
               << indices.size() << endl;
          cout << "Final region size = " << R.size() << endl;
          //getKey();
        }
        // TODO: for some reason, we have to clear the array of indices...
        // I have not figured out why...
        // The values read in each triple are strange as well.
        if (!indices.empty())
          indices.clear();
        break;
      }

    }
  }

  // This initializes an empty region $R$ heuristically with a triple of matches
  // $t = (m_i)_{1 \leq i \leq 3$.
  bool GrowRegion::buildSeedTriple(size_t t[3], const RegionBoundary& dR) const
  {
    if (dR.empty())
      return false;
    if (verbose_)
      printStage("Choose seed triple heuristically");
    t[0] = seed_;
    t[1] = dR.begin().index();    
    RegionBoundary::const_iterator tt = dR.begin();
    for (++tt; tt != dR.end(); ++tt)
    {
      t[2] = tt.index();
      if (!isDegenerate(t))
        return true;
    }
    return false;
  }

  // ======================================================================== //
  // Subroutine member functions
  void GrowRegion::updateBoundary(RegionBoundary& dR,
                                  const Region& R,
                                  size_t m)
  {    
    // Sanity check.
    if (m > M().size())
    {
      cerr << "FATAL ERROR in N_K access: ";
      cerr << "index "<< m << " is out of bounds!" << endl;
    }

    // Erase the added match $M_[m]$
    dR.erase(m);

    // Update the region boundary.
    const vector<size_t>& N_K_m = N_K(m);
    for (size_t j = 0; j != N_K_m.size(); ++j)
    {
      size_t ind_m_j = N_K_m[j];
      // When updating, make sure the candidate match $m_j$ is not in the 
      // region $R$. Otherwise, the region growing may loop forever.
      if (!R.find(ind_m_j) && 
          very_spurious_.find(ind_m_j) == very_spurious_.end())
        dR.insert(ind_m_j);
    }

    // Eliminate again spurious correspondences if they still are in the 
    // region boundary $\partial R$.
    vector<size_t> candidates(dR.size());
    vector<int> spurious(dR.size());
    size_t i = 0;
    RegionBoundary::iterator c = dR.begin();
    for ( ; c != dR.end(); ++c, ++i)
    {
      candidates[i] = c.index();
      spurious[i] = 0;
    }
    for (size_t i = 0; i != spurious.size(); ++i)
      if (very_spurious_.find(candidates[i]) != very_spurious_.end())
        spurious[i] = 1;
    for (size_t i = 0; i != spurious.size(); ++i)
      if (spurious[i] == 1)
        dR.erase(candidates[i]);
  }

  void GrowRegion::updateRegionAndBoundary(Region& R,
                                           RegionBoundary& dR,
                                           size_t m)
  {
    if (verbose_)
      cout << "Updating $R$ and $\\partial R$ by inserting M_["<< m << "]:\n"
           << M(m) << endl;
    R.insert(m);
    updateBoundary(dR, R, m);
  }

  vector<size_t> GrowRegion::get_N_K_m_cap_R(size_t m, const Region& R)
  {
    vector<size_t> N_K_cap_R;
    for (size_t j = 0; j != N_K(m).size(); ++j)
      if (R.find(N_K(m)[j]))
        N_K_cap_R.push_back(N_K(m)[j]);
    return N_K_cap_R;
  }

  bool GrowRegion::findTriple(size_t t[3],
                              const vector<size_t>& N_K_m_cap_R,
                              const PairWiseDrawer *pDrawer) const
  {
    // Safety check. Otherwise we cannot construct any candidate triple...
    if (N_K_m_cap_R.size() < 3)
    {
      NOTIFY_CANNOT_CONSTRUCT_N_k;
      return false;
    }
    DISPLAY_N_k;

    // Triple loop to find a good triple.
    for (size_t a = 0; a != N_K_m_cap_R.size(); ++a)
    {
      t[0] = N_K_m_cap_R[a];
      for (size_t b = a+1; b != N_K_m_cap_R.size(); ++b)
      {
        t[1] = N_K_m_cap_R[b];
        for (size_t c = b+1; c != N_K_m_cap_R.size(); ++c)
        {
          t[2] = N_K_m_cap_R[c];
          if (!isDegenerate(t))
          {
            DISPLAY_NON_DEGENERATE_TRIPLE;
            return true;
          }
        }
      }
    }
    PAUSE_SEED_TRIPLE_SEARCH;
    return false;
  }

  bool GrowRegion::findTriple(size_t t[3], size_t m, const Region& R,
                              const PairWiseDrawer *pDrawer)
  {
    // Get the subset of matches $\mathcal{N}_K(m) \cap R$.
    vector<size_t> N_K_m_cap_R(get_N_K_m_cap_R(m, R));
    return findTriple(t, N_K_m_cap_R);
  }

  bool GrowRegion::isDegenerate(size_t t[3]) const
  {
    Vector2d x[3], y[3];
    for (int i = 0; i < 3; ++i)
    {
      x[i] = M(t[i]).posX().cast<double>();
      y[i] = M(t[i]).posY().cast<double>();
    }
    return (params_.isFlat(x) || params_.isFlat(y));
  }

  bool GrowRegion::overlap(const size_t q[4], const Region& R) const
  {
    int count = 0;
    for (int i = 0; i < 4; ++i)
      if (R.find(q[i]))
        ++count;
    return count >= 3;
  }

  bool GrowRegion::overlap(vector<size_t>& regionIndices,
                           const vector<Region>& RR,
                           const size_t q[4]) const
  {
    if (!regionIndices.empty())
      regionIndices.clear();
    bool result = false;
    for (size_t i = 0; i != RR.size(); ++i)
    {
      if (overlap(q, RR[i]))
      {
        result = true;
        regionIndices.push_back(i);
      }
    }

    return result;        
  }

  // ======================================================================== //
  // Affine consistency test functions.
  bool GrowRegion::affineConsistent(const size_t q[4], int& very_spurious,
                                    const PairWiseDrawer *pDrawer) const
  {
    Match m[4];
    for (int i = 0; i < 4; ++i)
      m[i] = M(q[i]);

    for (int i = 0; i < 4; ++i)
    {
      // Compute the local affinity.
      Matrix3f phi, inv_phi;
      phi = affinityFromXToY(m[i], m[(i+1)%4], m[(i+2)%4]);
      inv_phi = phi.inverse();

//#define IT_MAKES_SENSE_TO_DILATE_KEYPOINTS
#ifdef IT_MAKES_SENSE_TO_DILATE_KEYPOINTS
      // Compute the transformed keypoints.
      OERegion x(m[(i+3)%4].x());
      OERegion y(m[(i+3)%4].y());
      double ab1 = sqrt(1./double(x.feat().shapeMat().determinant()));
      double ab2 = sqrt(1./double(y.feat().shapeMat().determinant()));
      double ab = max(ab1, ab2);
      double r = 20.;
      double lambda2 = ab/(r*r);
      x.feat().shapeMat() *= lambda2;
      y.feat().shapeMat() *= lambda2;
#else
      const OERegion& x = m[(i+3)%4].x();
      const OERegion& y = m[(i+3)%4].y();
#endif

#define I_KNOW_THE_ACCURACY_OF_LOCAL_AFFINITY
#ifdef I_KNOW_THE_ACCURACY_OF_LOCAL_AFFINITY
      // Don't do any ellipse transform first, if it turns out that we have a 
      // gross outliers.
      const Point2f& cx = x.center();
      const Point2f& cy = y.center();
      Point2f phi_cx(apply(phi, cx));
      Point2f inv_phi_cy(apply(inv_phi, cy));
      double sd1 = (cx-inv_phi_cy).squaredNorm(); // squared distance 1
      double sd2 = (cy-phi_cx).squaredNorm();     // squared distance 2
      double distThres = 15.f;
      double distThres2 = distThres*distThres;
      if (sd1 > distThres2 || sd2 > distThres2)
      {
        very_spurious = (i==0) ? 1 : 0;
        return false;
      }
#endif

      // Otherwise compute intersection area of intersecting ellipses.
      OERegion phi_x(transformOERegion(x,phi));
      OERegion inv_phi_y(transformOERegion(y,inv_phi));
      // Get the original ellipses.
      Ellipse S_x(ellipseFromOERegion(x));
      Ellipse S_y(ellipseFromOERegion(y));
      // Compute the transformed ellipses.
      Ellipse phi_S_x(ellipseFromOERegion(phi_x));
      Ellipse inv_phi_S_y(ellipseFromOERegion(inv_phi_y));
      // intersection/union area ratio.
      double overlapRatio[2] = {
        analyticJaccardSimilarity(phi_S_x, S_y),
        analyticJaccardSimilarity(S_x, inv_phi_S_y)
      };

      // Uncomment this only for debugging
      //checkLocalAffineConsistency(x,y,phi_x, inv_phi_y, overlapRatio, m, pDrawer);

      double overlapRatioT = /*m[i].featX().type() == OERegion::DoG ? 0.2 :*/ 0.4;

      if (overlapRatio[0] < overlapRatioT || overlapRatio[1] < overlapRatioT)
        return false;
    }

    return true;
  }

  // ======================================================================== //
  // Visual debugging member functions.
  void GrowRegion::checkRegion(const Region& R,
                               const PairWiseDrawer *pDrawer,
                               bool pause) const
  {
    if (verbose_)
      cout << "Check Region" << endl;
    R.view(M(), *pDrawer, Blue8);
    if (pause)
      getKey();
  }

  void GrowRegion::checkRegionBoundary(const RegionBoundary& dR,
                                       const PairWiseDrawer *pDrawer,
                                       bool pause) const
  {
    if (verbose_)
      cout << "Check Region Boundary" << endl;
    dR.view(*pDrawer);
    if (pause)
      getKey();
  }

  void GrowRegion::checkGrowingState(const Region& R, const RegionBoundary& dR,
                                     const PairWiseDrawer *pDrawer,
                                     bool pause) const
  {
    pDrawer->displayImages();
    checkRegionBoundary(dR, pDrawer);
    checkRegion(R, pDrawer);
    if (verbose_)
      cout << R << endl;
    if (pause)
      getKey();
  }

  void
  GrowRegion::
  checkLocalAffineConsistency(const OERegion& x, const OERegion& y,
                              const OERegion& phi_x, const OERegion& inv_phi_y,
                              const double overlapRatio[2],
                              const Match m[4],
                              const PairWiseDrawer *pDrawer) const
  {
    if (pDrawer)
    {
      // Check visually.
      Match rescaled_m;
      rescaled_m.ptrX() = &x;
      rescaled_m.ptrY() = &y;
      // Check visually.
      Match phi_m;
      phi_m.ptrX() = &inv_phi_y;
      phi_m.ptrY() = &phi_x;
      // Verbose
      cout << "Precision of Match Triple" << endl;
      for (int i = 0; i < 2; ++i)
        cout << "ratio[" << i+1 << "] = " << overlapRatio[i] << endl;
      pDrawer->displayImages();
      for (int i = 0; i < 3; ++i)
        pDrawer->drawMatch(m[i], Green8);
      pDrawer->drawMatch(m[3], Cyan8);
      pDrawer->drawMatch(rescaled_m, Magenta8);
      if (overlapRatio[0] < 0.2 || overlapRatio[1] < 0.2)
        pDrawer->drawMatch(phi_m, Red8);
      else
        pDrawer->drawMatch(phi_m, Yellow8);
      getKey();
    }
  }

  // ======================================================================== //
  // Functions for performance analysis.
  void GrowRegion::analyzeQuadruple(const size_t q[4],
                                    RegionGrowingAnalyzer *pAnalyzer)
  {
    if (verbose_)
      cout << "Analyzing quadruple" << endl;
    if (pAnalyzer)
    {
      for (int i = 0; i < 4; ++i)
      {
        size_t t[3] = { q[i], q[(i+1)%4], q[(i+2)%4] };
        size_t mm = q[(i+3)%4];
        if (pAnalyzer->isOfInterest(mm))
          pAnalyzer->analyzeQuad(t, mm);
      }
    }
  }

} /* namespace DO */
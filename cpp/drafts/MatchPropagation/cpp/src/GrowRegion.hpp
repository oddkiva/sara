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

#pragma once

#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Geometry.hpp>

#include "DynamicMatchGraph.hpp"
#include "GrowthParams.hpp"
#include "Region.hpp"
#include "RegionBoundary.hpp"
#include "RegionGrowingAnalyzer.hpp"


namespace DO { namespace Sara {

  class DO_SARA_EXPORT GrowRegion
  {
  public: /* interface. */
    //! Constructor
    GrowRegion(size_t seed, DynamicMatchGraph& g, const GrowthParams& params,
               bool verbose = false);

    //! The main method.
    Region operator()(size_t max_region_size = std::numeric_limits<size_t>::max(),
                      const PairWiseDrawer* drawer = nullptr,
                      RegionGrowingAnalyzer* analyzer = nullptr);

    //! Alternative main method which however stops growing a region if it ever
    //! overlaps with prior grown region.
    std::pair<Region, std::vector<size_t>>
    operator()(const std::vector<Region>& RR,
               size_t max_region_size = std::numeric_limits<size_t>::max(),
               const PairWiseDrawer* drawer = nullptr,
               RegionGrowingAnalyzer* analyzer = nullptr);

    //! For debugging purposes, activate the debug flag.
    void set_verbose(bool on = true)
    {
      _verbose = on;
    }

    const std::set<size_t>& get_spurious_matches() const
    {
      return _very_spurious;
    }

  private: /* important member functions. */
    //! A region growing is done in two steps.
    //! 1. Try initializing the region with an affine-consistent quadruple.
    bool initialize_affine_quadruple(Region& R, RegionBoundary& dR,
                                     const PairWiseDrawer* drawer = 0);

    //! 2. If initialization is successful, grow the region.
    //!    However, the growing process stops if it intersects with other grown
    //!    regions.
    //!    This returns a set of indices corresponding to regions with which
    //!    the region being grown $R$ intersects.
    void grow(const std::vector<Region>& RR, Region& R, RegionBoundary& dR,
              std::vector<size_t>& overlap,
              size_t max_allowed_size = std::numeric_limits<size_t>::max(),
              const PairWiseDrawer* drawer = nullptr,
              RegionGrowingAnalyzer* analyzer = nullptr);

    /*! The method initializes an empty region $R$ heuristically with
     *  a triple of matches $t = (m_i)_{1 \leq i \leq 3$.
     *  The method is called by the method:
     *  bool GrowRegion::initAffQuad(Region& R, RegionBoundary& dR,
     *                               const PairWiseDrawer *drawer = 0) const
     *  which then finds a match $m in \bigcup_{m \in t} \mathcal{N}_K(m)$ such
     *  that $q = (t, m)$ forms an affine-consistent quadruple.
     */
    bool build_seed_triple(size_t t[3], const RegionBoundary& dR) const;

  private: /* subroutine member functions. */
    //! Update functions when we find an affine-consistent match $m$.
    void update_boundary(RegionBoundary& dR, const Region& R, size_t m);

    void update_region_and_boundary(Region& R, RegionBoundary& dR, size_t m);

    /*! This method returns the set of matches that are neighbors of match $m$
     *  and that are also in the region $R$, i.e., $\mathcal{N}_K(m) \cap R$.
     */
    std::vector<size_t> get_N_K_m_cap_R(size_t m, const Region& R);

    /*! For a given match $m$, this method finds a non degenerate triple of
     *  matches $t = (m_i)_{1 \leq i \leq 3$ such that
     *  $t \in \mathcal{N}_K(m)^3$.
     */
    bool find_triple(size_t t[3], const std::vector<size_t>& N_K_m_cap_R,
                     const PairWiseDrawer* drawer = nullptr) const;

    //! DEPRECATED:
    //! Don't use it anymore as it is not suitable for parallelization.
    bool find_triple(size_t t[3], size_t m, const Region& R);

    //! Check that the triple of match $t$ is not degenerate.
    bool is_degenerate(size_t t[3]) const;

    //! $q$ denotes a quadruple of matches $(m_i)_{1 \leq i \leq 4$.
    bool affine_consistent(const size_t q[4], int& very_spurious,
                           const PairWiseDrawer* drawer = nullptr) const;

    /*! A quadruple of matches $q = (m_i)_{1 \leq i \leq 4}$ overlaps with a
     *  region $R$ if at least three matches of $q$ are in the region $R$.
     *
     *  Note that we construct a quadruple of matches $q = (t,m)$ such that the
     *  triple $t \in \mathcal{N}_K(m)^3$, it makes sense to check that at least
     *  $3$ matches are in region $R$ in order to decide if a quadruple $q$
     *  overlaps with region $R$.
     *
     *  We recall that the triple of matches $t$ is constructed in the method:
     *  bool GrowRegion::findTriple(size_t t[3], size_t m, const Region& R,
     *                              const PairWiseDrawer *drawer = 0) const
     */
    bool overlap(const size_t q[4], const Region& R) const;

    bool overlap(std::vector<size_t>& indices, const std::vector<Region>& RR,
                 const size_t q[4]) const;

  private: /* visual debugging member functions. */
    void check_region(const Region& R, const PairWiseDrawer* drawer,
                      bool pause = false) const;

    void check_region_boundary(const RegionBoundary& dR,
                               const PairWiseDrawer* drawer,
                               bool pause = false) const;

    void check_region_growing_state(const Region& R, const RegionBoundary& dR,
                                    const PairWiseDrawer* drawer,
                                    bool pause = false) const;

    void check_local_affine_consistency(const OERegion& x, const OERegion& y,
                                        const OERegion& phi_x,
                                        const OERegion& inv_phi_y,
                                        const double overlapRatio[2],
                                        const Match m[4],
                                        const PairWiseDrawer* drawer = nullptr) const;

  private: /* functions for performance analysis. */
    void analyze_quadruple(const size_t q[4], RegionGrowingAnalyzer* analyzer);

  private: /* helper functions */
    const Match& M(size_t i) const
    {
      return G_.M(i);
    }

    const std::vector<size_t>& N_K(size_t i)
    {
      return G_.N_K(i);
    }

    const std::vector<Match>& M() const
    {
      return G_.M();
    }

  private:
    /*!
     * @brief 'seed_' is the index of the seed match.
     *
     * The seed match is the match from which we want to grow
     * maximally consistent region $R$ and the seed match is stored in the array
     * element 'g_.M(seed_)'.
     *
     */
    size_t _seed;

    //! @brief Dynamic graph of matches containing the set of initial matches
    //! $\mathcal{M}$.
    DynamicMatchGraph& G_;

    //! @brief Growth parameters.
    GrowthParams _params;

    //! @brief Verbose flag for debugging.
    bool _verbose;

    //! @brief EXPERIMENTAL.
    std::set<size_t> _very_spurious;
  };

}}  // namespace DO::Sara

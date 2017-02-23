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

// ========================================================================== //
//  Efficient and Scalable 4th-order Match Propagation
//  David Ok, Renaud Marlet, and Jean-Yves Audibert.
//  ACCV 2012, Daejeon, South Korea.
//
// Copyright (c) 2013. David Ok, Imagine (ENPC/CSTB).
// ========================================================================== //

#pragma once

#include "MatchNeighborhood.hpp"


namespace DO { namespace Sara { namespace Extensions {

  class DynamicMatchGraph
  {
  public:
    //! Constructor
    DynamicMatchGraph(const std::vector<Match>& M, size_t K, float rho_min)
      : M_(M)
      , K_(K)
      , rho_min_(rho_min)
      , compute_N_K_(M)
    {
      N_K_.resize(M.size());
      N_K_is_computed_.resize(M.size());

      for (auto i = 0u; i != N_K_.size(); ++i)
      {
        N_K_[i].reserve(1000u);
        N_K_is_computed_[i] = 0;
      }
    }

    //! Returns constant reference to set of matches $\mathcal{M}$.
    const std::vector<Match>& M() const
    {
      return M_;
    }

    //! Returns the number of initial matches.
    size_t size() const
    {
      return M_.size();
    }

    //! Returns constant reference to match $m_i$.
    const Match& M(size_t i) const
    {
      return M_[i];
    }

    //! Returns constant reference to the neighborhood of match $m_i$.
    const std::vector<size_t>& N_K(size_t i)
    {
      update_N_K(i);
      return N_K_[i];
    }

    /*! This is used by the method:
     *  const std::vector<size_t>& DynamicMatchGraph::N_K(size_t i)
     *  in order to dynamically update $\mathcal{N}_K(m_i)$.
     */
    void update_N_K(size_t i)
    {
      if (N_K_is_computed_[i] == 0)
      {
        N_K_[i] = compute_N_K_(i, K_, rho_min_ * rho_min_);
        N_K_is_computed_[i] = 1;
      }
    }

    void update_N_K(const std::vector<size_t>& indices)
    {
      std::vector<size_t> indices_to_update;
      indices_to_update.reserve(indices.size());
      for (size_t i = 0; i != indices.size(); ++i)
        if (N_K_is_computed_[indices[i]] == 0)
          indices_to_update.push_back(indices[i]);

      std::vector<std::vector<size_t>> N_K_indices;
      N_K_indices = compute_N_K_(indices_to_update, K_, rho_min_ * rho_min_);

      for (size_t i = 0; i != indices_to_update.size(); ++i)
      {
        N_K_[indices_to_update[i]] = N_K_indices[i];
        N_K_is_computed_[indices_to_update[i]] = 1;
      }
    }

    //! Warning: this is not computationally efficient. This method is
    //! only for performance study.
    void build_hat_N_Ks()
    {
      /*for (size_t i = 0; i != N_K_.size(); ++i)
        updateN_K(i);*/
      std::vector<size_t> indices(N_K_.size());
      for (size_t i = 0; i != N_K_.size(); ++i)
        indices[i] = i;
      updateN_K(indices);
      std::vector<std::vector<size_t>> hat_N_K(computeHatN_K(N_K_));
      N_K_.swap(hat_N_K);
    }

  private: /* data members. */
    //! @brief Set of initial matches $\mathcal{M}$.
    const std::vector<Match>& M_;

    //! @{
    //! @brief Connectivity parameters.
    size_t K_;
    float rho_min_;
    //! @}

    //! @brief Match neighborhoods $\mathcal{M}$.
    std::vector<std::vector<size_t>> N_K_;

    /*!
     *  @brief Array of flags.
     *
     *  For each match $m_i$ stored in array element 'M_[i]', the flag
     *  takes the following value:
     *  'N_K_is_computed_[i] == 1' if we have computed its neighborhood
     *                             $\mathcal{N}_K(m_i)$ previously.
     *  'N_K_is_computed_[i] == 0' otherwise.
     *   If the flag is '0' then we will have to compute the neighborhood
     *   $\mathcal{N}_K(m_i)$ when needed.
     */
    std::vector<char> N_K_is_computed_;

    //! @brief Match neighborhood compute functor.
    ComputeN_K compute_N_K_;
  };

} /* namespace Extensions */
} /* namespace Sara */
} /* namespace DO */

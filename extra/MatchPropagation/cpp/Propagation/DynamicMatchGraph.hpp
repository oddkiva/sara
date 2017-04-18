// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2017 David Ok <david.ok8@gmail.com>
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


namespace DO { namespace Sara { namespace extra {

  class DynamicMatchGraph
  {
  public:
    //! Constructor
    DynamicMatchGraph(const std::vector<Match>& M, size_t K, float rho_min)
      : _M{M}
      , _K{K}
      , _rho_min{rho_min}
      , _compute_N_K{M}
    {
      _N_K.resize(M.size());
      _N_K_is_computed.resize(M.size());

      for (auto i = 0u; i != _N_K.size(); ++i)
      {
        _N_K[i].reserve(1000u);
        _N_K_is_computed[i] = 0;
      }
    }

    //! Returns constant reference to set of matches $\mathcal{M}$.
    auto M() const -> const std::vector<Match>&
    {
      return _M;
    }

    //! Returns the number of initial matches.
    auto size() const -> std::size_t
    {
      return _M.size();
    }

    //! Returns constant reference to match $m_i$.
    auto M(size_t i) const -> const Match&
    {
      return _M[i];
    }

    //! Returns constant reference to the neighborhood of match $m_i$.
    auto N_K(size_t i) -> const std::vector<size_t>&
    {
      update_N_K(i);
      return _N_K[i];
    }

    /*! This is used by the method:
     *  const std::vector<size_t>& DynamicMatchGraph::N_K(size_t i)
     *  in order to dynamically update $\mathcal{N}_K(m_i)$.
     */
    auto update_N_K(size_t i) -> void
    {
      if (_N_K_is_computed[i] == 0)
      {
        _N_K[i] = _compute_N_K(i, _K, _rho_min * _rho_min);
        _N_K_is_computed[i] = 1;
      }
    }

    auto update_N_K(const std::vector<size_t>& indices) -> void
    {
      auto indices_to_update = std::vector<size_t>{};
      indices_to_update.reserve(indices.size());
      for (auto i = 0u; i != indices.size(); ++i)
        if (_N_K_is_computed[indices[i]] == 0)
          indices_to_update.push_back(indices[i]);

      auto N_K_indices =
          _compute_N_K(indices_to_update, _K, _rho_min * _rho_min);

      for (auto i = 0u; i != indices_to_update.size(); ++i)
      {
        _N_K[indices_to_update[i]] = N_K_indices[i];
        _N_K_is_computed[indices_to_update[i]] = 1;
      }
    }

    //! Warning: this is not computationally efficient. This method is
    //! only for performance study.
    void build_hat_N_Ks()
    {
      /*for (size_t i = 0; i != N_K_.size(); ++i)
        updateN_K(i);*/
      auto indices = std::vector<size_t>(_N_K.size());
      for (auto i = 0u; i != _N_K.size(); ++i)
        indices[i] = i;
      update_N_K(indices);
      auto hat_N_K = compute_Hat_N_K(_N_K);
      _N_K.swap(hat_N_K);
    }

  private: /* data members. */
    //! @brief Set of initial matches $\mathcal{M}$.
    const std::vector<Match>& _M;

    //! @{
    //! @brief Connectivity parameters.
    size_t _K;
    float _rho_min;
    //! @}

    //! @brief Match neighborhoods $\mathcal{M}$.
    std::vector<std::vector<size_t>> _N_K;

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
    std::vector<char> _N_K_is_computed;

    //! @brief Match neighborhood compute functor.
    ComputeN_K _compute_N_K;
  };

} /* namespace Extensions */
} /* namespace Sara */
} /* namespace DO */

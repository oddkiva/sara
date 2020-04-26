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

//! @addtogroup MatchPropagation
//! @{

//! @brief Debugging macros.
//! @{

#ifdef DEBUG
# define CHECK_STATE_AFTER_ADDING_SEED_MATCH \
if (drawer) \
{ \
  print_stage("Check the region $R$ and the region boundary $\\partial R$ after adding seed match"); \
  drawer->display_images(); \
  check_region_growing_state(R, dR, drawer, true); \
}
#else
# define CHECK_STATE_AFTER_ADDING_SEED_MATCH
#endif

#ifdef DEBUG
# define CHECK_INCREMENTAL_SEED_TRIPLE_CONSTRUCTION \
if (drawer) \
  check_region_growing_state(R, dR, drawer, true);
#else
# define CHECK_INCREMENTAL_SEED_TRIPLE_CONSTRUCTION
#endif

#ifdef DEBUG
#define CHECK_CANDIDATE_FOURTH_MATCH_FOR_SEED_QUADRUPLE \
if (drawer) \
{ \
  cout << "Trying M_[" << q[3] << "]\n" << M(q[3])  << endl; \
  check_region_growing_state(R, dR, drawer); \
  drawer->draw_match(*m, Red8); \
  get_key(); \
}
#else
# define CHECK_CANDIDATE_FOURTH_MATCH_FOR_SEED_QUADRUPLE
#endif

#ifdef DEBUG
# define CHECK_GROWING_STATE_AFTER_FINDING_AFFINE_SEED_QUADRUPLE \
check_region_growing_state(R, dR, drawer, true);
#else
# define CHECK_GROWING_STATE_AFTER_FINDING_AFFINE_SEED_QUADRUPLE
#endif

#ifdef DEBUG
# define CHECK_CANDIDATE_MATCH_AND_GROWING_STATE \
if (drawer) \
{ \
  cout << "Trying M_[" << q[m][3] << "]\n" << M(q[m][3])  << endl; \
  check_region_growing_state(R, dR, drawer); \
  drawer->draw_match(*m, Yellow8); \
  get_key(); \
}
#else
# define CHECK_CANDIDATE_MATCH_AND_GROWING_STATE
#endif

#ifdef DEBUG
# define NOTIFY_CANNOT_CONSTRUCT_N_k \
if (_verbose) \
  cout << "N_K_m_cap_R.size() < 3" << endl;
#else
# define NOTIFY_CANNOT_CONSTRUCT_N_k
#endif

#ifdef DEBUG
# define DISPLAY_N_k \
if (drawer) \
{ \
  cout << "Drawing N_k\n"; \
  drawer->display_images(); \
  for (size_t j = 0; j != N_K_m_cap_R.size(); ++j) \
  { \
    drawer->draw_match(M(N_K_m_cap_R[j]), Cyan8); \
  } \
}
#else
# define DISPLAY_N_k
#endif

#ifdef DEBUG
# define DISPLAY_NON_DEGENERATE_TRIPLE \
if (drawer) \
{ \
  drawer->display_images(); \
  for (size_t j = 0; j != N_K_m_cap_R.size(); ++j) \
    drawer->draw_match(M(N_K_m_cap_R[j]), Cyan8); \
  cout << "Found good triple" << endl; \
  cout << "t = { "; \
  for (int i = 0; i < 3; ++i) \
  { \
    drawer->draw_match(M(t[i]), Blue8); \
    cout << t[i]; \
    if (i<2) \
      cout << ", "; \
  } \
  cout << " }" << endl; \
  get_key(); \
}
#else
# define DISPLAY_NON_DEGENERATE_TRIPLE
#endif

#ifdef DEBUG
# define PAUSE_SEED_TRIPLE_SEARCH \
if (drawer) \
  get_key();
#else
# define PAUSE_SEED_TRIPLE_SEARCH
#endif

//! @}

//! @}

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

#ifdef DEBUG
# define CHECK_STATE_AFTER_ADDING_SEED_MATCH \
if (pDrawer) \
{ \
  printStage("Check the region $R$ and the region boundary $\\partial R$ after adding seed match"); \
  pDrawer->display_images(); \
  checkGrowingState(R, dR, pDrawer, true); \
}
#else
# define CHECK_STATE_AFTER_ADDING_SEED_MATCH
#endif

#ifdef DEBUG
# define CHECK_INCREMENTAL_SEED_TRIPLE_CONSTRUCTION \
if (pDrawer) \
  checkGrowingState(R, dR, pDrawer, true);
#else
# define CHECK_INCREMENTAL_SEED_TRIPLE_CONSTRUCTION
#endif

#ifdef DEBUG
#define CHECK_CANDIDATE_FOURTH_MATCH_FOR_SEED_QUADRUPLE \
if (pDrawer) \
{ \
  cout << "Trying M_[" << q[3] << "]\n" << M(q[3])  << endl; \
  checkGrowingState(R, dR, pDrawer); \
  pDrawer->draw_match(*m, Red8); \
  get_key(); \
}
#else
# define CHECK_CANDIDATE_FOURTH_MATCH_FOR_SEED_QUADRUPLE
#endif

#ifdef DEBUG
# define CHECK_GROWING_STATE_AFTER_FINDING_AFFINE_SEED_QUADRUPLE \
checkGrowingState(R, dR, pDrawer, true);
#else
# define CHECK_GROWING_STATE_AFTER_FINDING_AFFINE_SEED_QUADRUPLE
#endif

#ifdef DEBUG
# define CHECK_CANDIDATE_MATCH_AND_GROWING_STATE \
if (pDrawer) \
{ \
  cout << "Trying M_[" << q[3] << "]\n" << M(q[3])  << endl; \
  checkGrowingState(R, dR, pDrawer); \
  pDrawer->draw_match(*m, Yellow8); \
  get_key(); \
}
#else
# define CHECK_CANDIDATE_MATCH_AND_GROWING_STATE
#endif

#ifdef DEBUG
# define NOTIFY_CANNOT_CONSTRUCT_N_k \
if (verbose_) \
  cout << "N_k[" << m << "].size() < 3" << endl;
#else
# define NOTIFY_CANNOT_CONSTRUCT_N_k
#endif

#ifdef DEBUG
# define DISPLAY_N_k \
if (pDrawer) \
{ \
  cout << "Drawing N_k\n"; \
  pDrawer->display_images(); \
  checkRegion(R, pDrawer); \
  for (size_t j = 0; j != N_k.size(); ++j) \
  { \
    pDrawer->draw_match(M(N_k[j]), Cyan8); \
  } \
  pDrawer->draw_match(M(m), Yellow8); \
}
#else
# define DISPLAY_N_k
#endif

#ifdef DEBUG
# define DISPLAY_NON_DEGENERATE_TRIPLE \
if (pDrawer) \
{ \
  pDrawer->display_images(); \
  checkRegion(R, pDrawer); \
  for (size_t j = 0; j != N_k.size(); ++j) \
    pDrawer->draw_match(M(N_k[j]), Cyan8); \
  cout << "Found good triple" << endl; \
  cout << "t = { "; \
  for (int i = 0; i < 3; ++i) \
  { \
    pDrawer->draw_match(M(t[i]), Blue8); \
    cout << t[i]; \
    if (i<2) \
      cout << ", "; \
  } \
  cout << " }" << endl; \
  pDrawer->draw_match(M(m), Yellow8); \
  get_key(); \
}
#else
# define DISPLAY_NON_DEGENERATE_TRIPLE
#endif

#ifdef DEBUG
# define PAUSE_SEED_TRIPLE_SEARCH \
if (pDrawer) \
  get_key();
#else
# define PAUSE_SEED_TRIPLE_SEARCH
#endif

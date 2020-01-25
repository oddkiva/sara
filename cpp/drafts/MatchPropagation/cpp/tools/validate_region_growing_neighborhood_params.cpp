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

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FeatureMatching.hpp>
#include <DO/Sara/Graphics.hpp>

#include "EvaluateOutlierResistance.hpp"
#include "EvaluateQualityOfLocalAffineApproximation.hpp"
#include "Learn_P_f.hpp"
#include "MatchNeighborhood.hpp"
#include "StudyPerfWithHatN_K.hpp"
#include "Study_N_K_m.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace std;
using namespace DO;

// Dataset paths.
const string mikolajczyk_dataset_folder(
    "C:/Users/David/Desktop/doplusplus/trunk/DO/ValidateRegionGrowing");
const string folders[8] = {"bark",   "bikes", "boat", "graf",
                           "leuven", "trees", "ubc",  "wall"};
const string ext[4] = {".dog", ".haraff", ".hesaff", ".mser"};

bool performLearningOfP_f()
{
  // Extended Lowe's score threshold.
  float ell = 1.2f;
  // Parallelize these tasks.
  int num_dataset = 8;
  int num_feat_type = 4;
  int num_tasks = num_dataset * num_feat_type;
  bool approxInterEllArea = true;
#pragma omp parallel for
  for (int t = 0; t < num_tasks; ++t)
  {
    LearnPf* pLearnPf = 0;
    int dataset = t / num_feat_type;
    int feat_type = t % num_feat_type;
#pragma omp critical
    {
      pLearnPf = new LearnPf(mikolajczyk_dataset_folder, folders[dataset],
                             ext[feat_type], approxInterEllArea);
    }
    if (pLearnPf)
    {
      (*pLearnPf)(ell * ell);
      delete pLearnPf;
    }
  }

  return true;
}

bool performEvalQualLocalAff()
{
  // Extended Lowe's score threshold.
  float ell = 1.0f;
  // Parameters of region growths
  size_t num_growths = 5000;
  size_t K = 200;
  // size_t k = 10;
  double rho_min = 0.3;
  // Parallelize the following tasks.
  int num_dataset = 8;
  int num_feat_type = 4;
  int num_tasks = num_dataset * num_feat_type;

#pragma omp parallel for
  for (int t = 0; t < num_tasks; ++t)
  {
    EvalQualityOfLocalAffApprox* evalQualLocalAff = 0;
    int dataset = t / num_feat_type;
    int feat_type = t % num_feat_type;
#pragma omp critical
    {
      evalQualLocalAff = new EvalQualityOfLocalAffApprox(
          mikolajczyk_dataset_folder, folders[dataset], ext[feat_type]);
    }
    if (evalQualLocalAff)
    {
      (*evalQualLocalAff)(ell * ell, num_growths, K, rho_min);
      delete evalQualLocalAff;
    }
  }

  return true;
}

bool performEvalOutlierResistance()
{
  // Extended Lowe's score threshold.
  float ell = 1.0f;
  // Parameters of region growths
  size_t num_growths = 5000;
  size_t K = 200;
  size_t k = 10;
  double rho_min = 0.5;
  // Parallelize the following tasks.
  int num_dataset = 8;
  int num_feat_type = 4;
  int num_tasks = num_dataset * num_feat_type;

#pragma omp parallel for
  for (int t = 0; t < num_tasks; ++t)
  {
    EvalOutlierResistance* evalOutlierResistance = 0;
    int dataset = t / num_feat_type;
    int feat_type = t % num_feat_type;
#pragma omp critical
    {
      evalOutlierResistance = new EvalOutlierResistance(
          mikolajczyk_dataset_folder, folders[dataset], ext[feat_type]);
    }
    if (evalOutlierResistance)
    {
      (*evalOutlierResistance)(ell * ell, num_growths, K, k, rho_min);
      delete evalOutlierResistance;
    }
  }

  return true;
}

bool performStudyOnHatN_K()
{
  // Below: Mikolajczyk et al.'s parameter in their IJCV 2005 paper.
  float inlierThres = 1.5f;
  /*
   * (Continued)
   * Let (x,y) be a match. It is an inlier if it satisfies:
   * $$\| \mathbf{H} \mathbf{x} - \mathbf{y} \|_2 < 1.5 \ \textrm{pixels}$$
   *
   * where $\mathbf{H}$ is the ground truth homography.
   * 1.5 pixels is used in the above-mentioned paper.
   */
  // Extended Lowe's score threshold.
  float squaredEll = 1.0f;

  // Neighborhood parameters.
  vector<size_t> K;
  vector<double> squaredRhoMin;
  for (size_t i = 1; i <= 15; ++i)
    K.push_back(i * 10);
  for (size_t i = 1; i <= 10; ++i)
    squaredRhoMin.push_back(double(i) * 0.1);


  for (int i = 0; i < 8; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      Study_N_K_m study(mikolajczyk_dataset_folder, folders[i], ext[j]);

      for (size_t k = 0; k != K.size(); ++k)
        for (size_t l = 0; l != squaredRhoMin.size(); ++l)
          if (!study(inlierThres, squaredEll, K[k], squaredRhoMin[l]))
            return false;
    }
  }

  return true;
}

bool performStudyPerfWithHatN_K()
{
  // Extended Lowe's score threshold.
  float ell = 1.0f;
  // Parameters of region growths
  size_t num_growths = 1000;
  size_t K = 200;
  double rho_min = 0.3;
  // Parallelize the following tasks.
  //  int num_dataset = 8;
  int num_feat_type = 4;
  //  int num_tasks = num_dataset*num_feat_type;

  // omp_set_num_threads(1);

  //#pragma omp parallel for
  //  for (int t = 0; t < num_tasks; ++t)
  //  {
  //    StudyPerfWithHat_N_K *studyPerfOnHatN_K = 0;
  //    int dataset = t / num_feat_type;
  //    int feat_type = t%num_feat_type;
  //#pragma omp critical
  //    {
  //      studyPerfOnHatN_K = new StudyPerfWithHat_N_K(
  //        mikolajczyk_dataset_folder,
  //        folders[dataset], ext[feat_type]);
  //    }
  //    if (studyPerfOnHatN_K)
  //    {
  //      (*studyPerfOnHatN_K)(ell*ell, num_growths, K, rho_min);
  //      delete studyPerfOnHatN_K;
  //    }
  //  }

  for (int f = 0; f < num_feat_type; ++f)
  {
    for (int d = 7; d < 8; ++d)
    {
      StudyPerfWithHat_N_K studyPerfOnHatN_K(mikolajczyk_dataset_folder,
                                             folders[d], ext[f]);
      studyPerfOnHatN_K(ell * ell, num_growths, K, rho_min);
    }
  }
  return true;
}

bool performStudyOnDynAffDrivenN_K()
{
  return true;
}

// Main.
int main()
{
  // DONE
  // performLearningOfP_f();
  // performEvalQualLocalAff();
  // performEvalOutlierResistance();
  // TODO
  // performStudyOnHatN_K();
  performStudyPerfWithHatN_K();
  // performStudyOnDynAffDrivenN_K();
  return 0;
}

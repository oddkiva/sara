#pragma once

#include <vector>


namespace DO::Sara {

  struct BundleAdjustmentProblem
  {
    int num_cameras;
    int num_points;
    int num_observations;
    int num_parameters;

    std::vector<int> point_index;
    std::vector<int> camera_index;
    std::vector<double> observations;
    std::vector<double> parameters;

    double* mutable_cameras()
    {
      return parameters.data();
    }

    double* mutable_points()
    {
      return parameters.data() + 9 * num_cameras;
    }

    double* mutable_camera_for_observation(int i)
    {
      return mutable_cameras() + camera_index[i] * 9;
    }

    double* mutable_point_for_observation(int i)
    {
      return mutable_points() + point_index[i] * 3;
    }

    auto populate(int num_observations, int num_points, int num_cameras)
    {
      // Observation = 2D points.
      // Point = 3D points.
      // Camera = 9D vector. (rotation, translation, focal, l1, l2).
      //
      // Parameters
      point_index.resize(num_observations);
      camera_index.resize(num_observations);
      observations.resize(2 * num_observations);
      num_parameters = 9 * num_cameras + 3 * num_points;
      parameters.resize(num_parameters);
    }

    //  for (int i = 0; i < num_observations; ++i)
    //  {
    //    FscanfOrDie(fptr, "%d", camera_index_ + i);
    //    FscanfOrDie(fptr, "%d", point_index_ + i);
    //    for (int j = 0; j < 2; ++j)
    //      FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
    //  }

    //  for (int i = 0; i < num_parameters_; ++i)
    //    FscanfOrDie(fptr, "%lf", parameters_ + i);
    //}
  };

} /* namespace DO::Sara */

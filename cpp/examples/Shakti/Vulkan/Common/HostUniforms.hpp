#pragma once

#include <Eigen/Geometry>


struct ModelViewProjectionStack
{
  ModelViewProjectionStack()
  {
    model.setIdentity();
    view.setIdentity();
  }

  Eigen::Transform<float, 3, Eigen::Projective> model;
  Eigen::Transform<float, 3, Eigen::Projective> view;
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
};

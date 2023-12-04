#pragma once

#include <Eigen/Geometry>


struct ModelViewProjectionStack
{
  ModelViewProjectionStack()
  {
    model.setIdentity();
    view.setIdentity();
    projection.setIdentity();
  }

  Eigen::Transform<float, 3, Eigen::Projective> model;
  Eigen::Transform<float, 3, Eigen::Projective> view;
  Eigen::Transform<float, 3, Eigen::Projective> projection;
};

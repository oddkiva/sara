#pragma once

#include <Eigen/Dense>


auto decompose_H_RQ_factorization(const Eigen::Matrix3d& H,
                                  const Eigen::Matrix3d& K,
                                  std::vector<Eigen::Matrix3d>& Rs,
                                  std::vector<Eigen::Vector3d>& ts,
                                  std::vector<Eigen::Vector3d>& ns) -> void;

auto decompose_H_faugeras(const Eigen::Matrix3d& H, const Eigen::Matrix3d& K,
                          std::vector<Eigen::Matrix3d>& Rs,
                          std::vector<Eigen::Vector3d>& ts,
                          std::vector<Eigen::Vector3d>& ns) -> void;

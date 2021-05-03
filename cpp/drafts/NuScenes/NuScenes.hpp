// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2021-present David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#pragma once

#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Eigen>


/*!
 *  The typical pattern regarding the data organization in NuScenes tables is as
 *  follows:
 *  - Each row of a any table is referenced by a UUID termed as "token" in
 *    NuScenes terminology.
 *    From a data structure point of view, each row can be stored as a hash
 *    table.
 *  - Each table can also be viewed as a doubly-linked list because of "prev"
 *    and "next" fields.
 */
struct NuScenes
{
  std::string version = "v1.0-mini";
  std::string dataroot = "/data/sets/nuscenes";
  bool verbose = true;
  std::vector<std::string> table_names = {
      "category", "attribute",   "visibility",
      "instance", "sensor",      "calibrated_sensor",
      "ego_pose", "log",         "scene",
      "sample",   "sample_data", "sample_annotation",
      "map"  //
  };

  using Token = std::string;

  struct Sample
  {
    Token prev;
    Token next;

    //! @brief The scene in which the sample is acquired.
    Token scene_token;

    //! @brief A sample is also referenced by a timestamp.
    std::uint64_t timestamp;
  };

  // For a given sample, we index the following data:
  // - images acquired from the 6 cameras (closest to this sample timestamp).
  // - point cloud acquired from the lidar device (closest to this sample
  //   timestamp),
  // - velocities acquired from the radar device (closest to this sample
  //   timestamp)
  struct SampleData
  {
    Token prev;
    Token next;

    Token sample_token;
    Token ego_pose_token;
    Token calibrated_sensor_token;

    std::uint64_t timestamp;
    std::string fileformat;
    bool is_key_frame;
    std::optional<int> width;
    std::optional<int> height;

    //! @brief An image file, lidar point cloud file or radar velocity file.
    std::string filename;

    auto is_image() const
    {
      return width.has_value() && height.has_value() && fileformat == "jpg";
    }
  };

  struct SampleAnnotation
  {
    Token prev;
    Token next;

    Token sample_token;
    Token instance_token;
    int visibility_token;
    std::vector<Token> attributes_tokens;

    // The 3D bounding box.
    Eigen::Vector3f size;
    Eigen::Vector3f translation;
    Eigen::Quaternionf rotation;

    // The lidar and radar metadata.
    int num_lidar_pts;
    int num_radar_pts;
  };

  struct Category
  {
    std::string name;
    std::string description;
    int index;
  };

  struct EgoPose
  {
    Eigen::Quaternionf rotation;
    Eigen::Vector3f translation;

    //! @brief A sample is also referenced by a timestamp.
    std::uint64_t timestamp;
  };

  struct CalibratedSensor
  {
    Eigen::Quaternionf rotation;
    Eigen::Vector3f translation;
    std::optional<Eigen::Matrix3f> calibration_matrix;
    Token sensor_token;
  };

  // An object instance should be understood as the same object that is seen
  // multiple times in multiple image frames, or point cloud data and sensor
  // data.
  struct Instance
  {
    Token category_token;
    int number_of_annotations;
    Token first_annotation_token;
    Token last_annotation_token;
  };

  struct Visibility
  {
    std::string description;
    std::string level;
  };

  struct Attribute
  {
    std::string name;
    std::string description;
  };

  struct Sensor
  {
    std::string channel;
    std::string modality;
  };

  std::unordered_map<Token, Sample> sample_table;
  std::unordered_map<Token, SampleData> sample_data_table;
  std::unordered_map<Token, SampleAnnotation> sample_annotation_table;
  std::unordered_map<Token, Category> category_table;
  std::unordered_map<Token, EgoPose> ego_pose_table;
  std::unordered_map<Token, CalibratedSensor> calibrated_sensor_table;
  std::unordered_map<Token, Instance> instance_table;
  std::unordered_map<Token, Visibility> visibility_table;
  std::unordered_map<Token, Attribute> attribute_table;
  std::unordered_map<Token, Sensor> sensor_table;

  NuScenes(const std::string& version_,   //
           const std::string& dataroot_,  //
           bool verbose_/*,               //
           float map_resolution = 0.1*/);

  auto load_sample_table() -> void;

  auto load_sample_data_table() -> void;

  auto load_sample_annotation_table() -> void;

  auto load_category_table() -> void;

  auto load_ego_pose_table() -> void;

  auto load_calibrated_sensor_table() -> void;

  auto load_instance_table() -> void;

  auto load_visibility_table() -> void;

  auto load_attribute_table() -> void;

  auto load_sensor_table() -> void;

  template <typename T>
  auto filter_by_sample_token(const std::unordered_map<Token, T>& table, const Token& value) const
  {
    auto rows = std::vector<T>{};
    std::for_each(table.cbegin(), table.cend(), [&](const auto& row) {
      if (row.second.sample_token == value)
        rows.emplace_back(row.second);
    });
    return rows;
  }

  auto get_data_path(const SampleData& data) const -> std::string;

  auto get_ego_pose(const SampleData& data) const -> const EgoPose&;

  auto get_calibrated_sensor(const SampleData& data) const
      -> const CalibratedSensor&;

  auto get_sensor(const CalibratedSensor& calibrated_sensor) const
      -> const Sensor&;
};

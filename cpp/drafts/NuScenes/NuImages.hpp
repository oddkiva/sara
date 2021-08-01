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


struct NuImages
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
    Token log_token;
    Token key_camera_token;

    //! @brief A sample is also referenced by a timestamp.
    std::uint64_t timestamp;
  };

  // For a given sample, we index the images acquired from the 6 cameras
  // (closest to this sample timestamp).
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
    int width;
    int height;

    //! @brief An image file, lidar point cloud file or radar velocity file.
    std::string filename;
  };

  struct ObjectAnnotation
  {
    Token sample_data_token;
    Token category_token;
    std::vector<Token> attribute_tokens;

    // The 2D bounding box.
    Eigen::Vector4f bbox;

    // TODO: mask.
  };

  struct Category
  {
    std::string name;
    std::string description;
  };

  struct EgoPose
  {
    Eigen::Quaternionf rotation;
    Eigen::Vector3f translation;

    //! IMU data.
    //!
    //! @brief Gyroscope.
    Eigen::Vector3f rotation_rate;
    //! @brief Acceleration.
    Eigen::Vector3f acceleration;

    float speed;

    //! @brief A sample is also referenced by a timestamp.
    std::uint64_t timestamp;
  };

  struct CalibratedSensor
  {
    // This is what I think it is.
    struct CameraDistortion
    {
      Eigen::Vector3f k;
      Eigen::Vector2f p;
    };

    Eigen::Quaternionf rotation;
    Eigen::Vector3f translation;
    Eigen::Matrix3f calibration_matrix;
    CameraDistortion camera_distortion;

    Token sensor_token;
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

  struct Log
  {
    std::string logfile;
    std::string location;
    std::string vehicle;
    std::string date_captured;
  };

  std::unordered_map<Token, Sample> sample_table;
  std::unordered_map<Token, SampleData> sample_data_table;
  std::unordered_map<Token, ObjectAnnotation> object_annotation_table;
  std::unordered_map<Token, Category> category_table;
  std::unordered_map<Token, EgoPose> ego_pose_table;
  std::unordered_map<Token, CalibratedSensor> calibrated_sensor_table;
  std::unordered_map<Token, Attribute> attribute_table;
  std::unordered_map<Token, Sensor> sensor_table;
  std::unordered_map<Token, Log> log_table;

  NuImages(const std::string& version_,   //
           const std::string& dataroot_,  //
           bool verbose_/*,               //
           float map_resolution = 0.1*/);

  auto load_sample_table() -> void;

  auto load_sample_data_table() -> void;

  auto load_object_annotation_table() -> void;

  auto load_category_table() -> void;

  auto load_ego_pose_table() -> void;

  auto load_calibrated_sensor_table() -> void;

  auto load_attribute_table() -> void;

  auto load_sensor_table() -> void;

  auto load_log_table() -> void;

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

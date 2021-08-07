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

#include <drafts/NuScenes/NuScenes.hpp>

#include <boost/filesystem.hpp>

#include <nlohmann/json.hpp>


static auto load_json(const std::string& dataroot, const std::string& version,
                      const std::string& table_name) -> nlohmann::json
{
  namespace fs = boost::filesystem;

  const auto table_json_filepath =
      fs::path{dataroot} / version / (table_name + ".json");

  std::ifstream table_json_file{table_json_filepath.string()};
  if (!table_json_file)
    throw std::runtime_error{"Could not open JSON file!"};
  auto table_json = nlohmann::json{};
  table_json_file >> table_json;

  return table_json;
}


NuScenes::NuScenes(const std::string& version_,   //
           const std::string& dataroot_,  //
           bool verbose_/*,               //
           float map_resolution = 0.1*/)
    : version{version_}
    , dataroot{dataroot_}
    , verbose{verbose_}
{
  // List of annotations.
  load_sample_table();

  // Sensor data.
  load_sample_data_table();
  load_sensor_table();

  // 3D bounding boxes.
  load_sample_annotation_table();

  // Egomotion.
  load_ego_pose_table();
  load_calibrated_sensor_table();

  // Various taxonomy of annotation attributes.
  load_category_table();
  load_instance_table();
  load_visibility_table();
  load_attribute_table();
}

auto NuScenes::load_sample_table() -> void
{
  const auto sample_json = load_json(dataroot, version, "sample");
  for (const auto& j : sample_json)
    sample_table[j["token"]] = {
        j["prev"],
        j["next"],
        j["scene_token"],
        j["timestamp"].get<std::uint64_t>()  //
    };
}

auto NuScenes::load_sample_data_table() -> void
{
  const auto sample_data_json = load_json(dataroot, version, "sample_data");
  for (const auto& j : sample_data_json)
  {
    sample_data_table[j["token"]] = {
        j["prev"],
        j["next"],
        j["sample_token"],
        j["ego_pose_token"],
        j["calibrated_sensor_token"],
        j["timestamp"].get<std::uint64_t>(),
        j["fileformat"],
        j["is_key_frame"].get<bool>(),
        j["width"].get<int>(),
        j["height"].get<int>(),
        j["filename"]};

    if (sample_data_table[j["token"]].width == 0 ||
        sample_data_table[j["token"]].height == 0)
    {
      sample_data_table[j["token"]].width = std::nullopt;
      sample_data_table[j["token"]].height = std::nullopt;
    }
  }
}

auto NuScenes::load_sample_annotation_table() -> void
{
  const auto sample_annotation_json =
      load_json(dataroot, version, "sample_annotation");
  for (const auto& j : sample_annotation_json)
  {
    auto sample_annotation = SampleAnnotation{};
    sample_annotation.prev = j["prev"];
    sample_annotation.next = j["next"];
    sample_annotation.sample_token = j["sample_token"];
    sample_annotation.visibility_token =
        std::stoi(j["visibility_token"].get<std::string>());
    sample_annotation.instance_token = j["instance_token"];
    for (const auto& jattr : j["attribute_tokens"])
      sample_annotation.attributes_tokens.emplace_back(
          jattr.get<std::string>());

    for (auto i = 0; i < 3; ++i)
      sample_annotation.translation(i) = j["translation"][i].get<float>();

    for (auto i = 0; i < 3; ++i)
      sample_annotation.size(i) = j["size"][i].get<float>();

    sample_annotation.rotation = Eigen::Quaternionf{
        j["rotation"][0].get<float>(), j["rotation"][1].get<float>(),
        j["rotation"][2].get<float>(), j["rotation"][3].get<float>()  //
    };

    sample_annotation.num_lidar_pts = j["num_lidar_pts"].get<int>();
    sample_annotation.num_radar_pts = j["num_radar_pts"].get<int>();

    sample_annotation_table[j["token"]] = sample_annotation;
  }
}

auto NuScenes::load_category_table() -> void
{
  const auto category_json = load_json(dataroot, version, "category");

  category_table.reserve(category_json.size());
  for (const auto& j : category_json)
    category_table[j["token"]] = {j["name"], j["description"], j["index"]};
}

auto NuScenes::load_ego_pose_table() -> void
{
  const auto ego_pose_json = load_json(dataroot, version, "ego_pose");

  ego_pose_table.reserve(ego_pose_json.size());
  for (const auto& j : ego_pose_json)
  {
    auto ego_pose = EgoPose{};

    for (auto i = 0; i < 3; ++i)
      ego_pose.translation(i) = j["translation"][i].get<float>();

    ego_pose.rotation = Eigen::Quaternionf{
        j["rotation"][0].get<float>(), j["rotation"][1].get<float>(),
        j["rotation"][2].get<float>(), j["rotation"][3].get<float>()  //
    };

    ego_pose.timestamp = j["timestamp"].get<std::uint64_t>();

    ego_pose_table[j["token"]] = ego_pose;
  }
}

auto NuScenes::load_calibrated_sensor_table() -> void
{
  const auto calibrated_sensor_json =
      load_json(dataroot, version, "calibrated_sensor");

  calibrated_sensor_table.reserve(calibrated_sensor_json.size());
  for (const auto& j : calibrated_sensor_json)
  {
    auto calibrated_sensor = CalibratedSensor{};

    for (auto i = 0; i < 3; ++i)
      calibrated_sensor.translation(i) = j["translation"][i].get<float>();

    calibrated_sensor.rotation = Eigen::Quaternionf{
        j["rotation"][0].get<float>(), j["rotation"][1].get<float>(),
        j["rotation"][2].get<float>(), j["rotation"][3].get<float>()  //
    };

    if (!j["camera_intrinsic"].empty())
    {
      auto K = Eigen::Matrix3f{};
      for (auto m = 0; m < 3; ++m)
        for (auto n = 0; n < 3; ++n)
          K(m, n) = j["camera_intrinsic"][m][n].get<float>();
      calibrated_sensor.calibration_matrix = K;
    }

    calibrated_sensor.sensor_token = j["sensor_token"].get<std::string>();

    calibrated_sensor_table[j["token"]] = calibrated_sensor;
  }
}

auto NuScenes::load_instance_table() -> void
{
  const auto instance_json = load_json(dataroot, version, "instance");

  instance_table.reserve(instance_json.size());
  for (const auto& j : instance_json)
  {
    auto instance = Instance{};

    instance_table[j["token"]] = {
        j["category_token"],
        j["nbr_annotations"].get<int>(),
        j["first_annotation_token"],
        j["last_annotation_token"]};
  }
}

auto NuScenes::load_visibility_table() -> void
{
  const auto visibility_json = load_json(dataroot, version, "visibility");

  visibility_table.reserve(visibility_json.size());
  for (const auto& j : visibility_json)
    visibility_table[j["token"]] = {
        j["description"],  //
        j["level"]               //
    };
}

auto NuScenes::load_attribute_table() -> void
{
  const auto attribute_json = load_json(dataroot, version, "attribute");

  attribute_table.reserve(attribute_json.size());
  for (const auto& j : attribute_json)
    attribute_table[j["token"]] = {
        j["name"],               //
        j["description"]  //
    };
}

auto NuScenes::load_sensor_table() -> void
{
  const auto sensor_json = load_json(dataroot, version, "sensor");

  sensor_table.reserve(sensor_json.size());
  for (const auto& j : sensor_json)
    sensor_table[j["token"]] = {
        j["channel"],   //
        j["modality"]  //
    };
}

auto NuScenes::get_data_path(const SampleData& data) const -> std::string
{
  namespace fs = boost::filesystem;
  return (fs::path{dataroot} / data.filename).string();
}

auto NuScenes::get_ego_pose(const SampleData& data) const -> const EgoPose&
{
  return ego_pose_table.at(data.ego_pose_token);
}

auto NuScenes::get_calibrated_sensor(const SampleData& data) const
    -> const CalibratedSensor&
{
  return calibrated_sensor_table.at(data.calibrated_sensor_token);
}

auto NuScenes::get_sensor(const CalibratedSensor& calibrated_sensor) const
    -> const Sensor&
{
  return sensor_table.at(calibrated_sensor.sensor_token);
}

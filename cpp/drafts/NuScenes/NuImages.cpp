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

#include <drafts/NuScenes/NuImages.hpp>

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

NuImages::NuImages(const std::string& version_,   //
                   const std::string& dataroot_,  //
                   bool verbose_/*,               //
                                  float map_resolution = 0.1*/)
: version{version_}
, dataroot{dataroot_}
, verbose{verbose_}
{
  // List of annotations.
  load_sample_table();

  // Image data.
  load_sample_data_table();
  load_sensor_table();

  // 2D bounding boxes.
  load_object_annotation_table();

  // Misc.
  load_log_table();

  // Egomotion.
  load_ego_pose_table();
  load_calibrated_sensor_table();

  // Various taxonomy of annotation attributes.
  load_category_table();
  load_attribute_table();
}

auto NuImages::load_sample_table() -> void
{
  const auto sample_json = load_json(dataroot, version, "sample");
  for (const auto& j : sample_json)
    sample_table[j["token"]] = {
        .log_token = j["log_token"],
        .key_camera_token = j["key_camera_token"],
        .timestamp = j["timestamp"].get<std::uint64_t>()  //
    };
}

auto NuImages::load_sample_data_table() -> void
{
  const auto sample_data_json = load_json(dataroot, version, "sample_data");
  for (const auto& j : sample_data_json)
    sample_data_table[j["token"]] = {
        .prev = j["prev"],
        .next = j["next"],
        .sample_token = j["sample_token"],
        .ego_pose_token = j["ego_pose_token"],
        .calibrated_sensor_token = j["calibrated_sensor_token"],
        .timestamp = j["timestamp"].get<std::uint64_t>(),
        .fileformat = j["fileformat"],
        .is_key_frame = j["is_key_frame"].get<bool>(),
        .width = j["width"].get<int>(),
        .height = j["height"].get<int>(),
        .filename = j["filename"]};
}

auto NuImages::load_object_annotation_table() -> void
{
  const auto object_annotation_json =
      load_json(dataroot, version, "object_ann");
  for (const auto& j : object_annotation_json)
  {
    auto object_annotation = ObjectAnnotation{};
    object_annotation.sample_data_token = j["sample_data_token"];
    object_annotation.category_token = j["category_token"];
    for (const auto& jattr : j["attribute_tokens"])
      object_annotation.attribute_tokens.emplace_back(jattr.get<std::string>());

    for (auto i = 0; i < 4; ++i)
      object_annotation.bbox(i) = j["bbox"][i].get<float>();

    object_annotation_table[j["token"]] = object_annotation;
  }
}

auto NuImages::load_category_table() -> void
{
  const auto category_json = load_json(dataroot, version, "category");

  category_table.reserve(category_json.size());
  for (const auto& j : category_json)
    category_table[j["token"]] = {j["name"], j["description"]};
}

auto NuImages::load_ego_pose_table() -> void
{
  const auto ego_pose_json = load_json(dataroot, version, "ego_pose");

  ego_pose_table.reserve(ego_pose_json.size());
  for (const auto& j : ego_pose_json)
  {
    auto ego_pose = EgoPose{};

    for (auto i = 0; i < 3; ++i)
      ego_pose.translation(i) = j["translation"][i].get<float>();

    ego_pose.rotation = Eigen::Quaternionf{
        j["rotation"][0].get<float>(),  //
        j["rotation"][1].get<float>(),  //
        j["rotation"][2].get<float>(),  //
        j["rotation"][3].get<float>()   //
    };

    for (auto i = 0; i < 3; ++i)
      ego_pose.acceleration(i) = j["acceleration"][i].get<float>();

    ego_pose.rotation_rate = Eigen::Vector3f{
        j["rotation_rate"][0].get<float>(),  //
        j["rotation_rate"][1].get<float>(),  //
        j["rotation_rate"][2].get<float>(),  //
    };

    ego_pose.speed = j["speed"].get<float>();

    ego_pose.timestamp = j["timestamp"].get<std::uint64_t>();

    ego_pose_table[j["token"]] = ego_pose;
  }
}

auto NuImages::load_calibrated_sensor_table() -> void
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

    for (auto m = 0; m < 3; ++m)
      for (auto n = 0; n < 3; ++n)
        calibrated_sensor.calibration_matrix(m, n) =
            j["camera_intrinsic"][m][n].get<float>();

    calibrated_sensor.sensor_token = j["sensor_token"].get<std::string>();

    for (auto i = 0; i < 3; ++i)
      calibrated_sensor.camera_distortion.k(i) =
          j["camera_distortion"][i].get<float>();

    for (auto i = 0; i < 2; ++i)
      calibrated_sensor.camera_distortion.p(i) =
          j["camera_distortion"][3 + i].get<float>();

    calibrated_sensor_table[j["token"]] = calibrated_sensor;
  }
}

auto NuImages::load_attribute_table() -> void
{
  const auto attribute_json = load_json(dataroot, version, "attribute");

  attribute_table.reserve(attribute_json.size());
  for (const auto& j : attribute_json)
    attribute_table[j["token"]] = {
        .name = j["name"],               //
        .description = j["description"]  //
    };
}

auto NuImages::load_sensor_table() -> void
{
  const auto sensor_json = load_json(dataroot, version, "sensor");

  sensor_table.reserve(sensor_json.size());
  for (const auto& j : sensor_json)
    sensor_table[j["token"]] = {
        .channel = j["channel"],   //
        .modality = j["modality"]  //
    };
}

auto NuImages::load_log_table() -> void
{
  const auto log_json = load_json(dataroot, version, "log");

  log_table.reserve(log_json.size());
  for (const auto& j : log_json)
    log_table[j["token"]] = {
        .logfile = j["logfile"],             //
        .location = j["location"],           //
        .vehicle = j["vehicle"],             //
        .date_captured = j["date_captured"]  //
    };
}

auto NuImages::get_data_path(const SampleData& data) const -> std::string
{
  namespace fs = boost::filesystem;
  return (fs::path{dataroot} / data.filename).string();
}

auto NuImages::get_ego_pose(const SampleData& data) const -> const EgoPose&
{
  return ego_pose_table.at(data.ego_pose_token);
}

auto NuImages::get_calibrated_sensor(const SampleData& data) const
    -> const CalibratedSensor&
{
  return calibrated_sensor_table.at(data.calibrated_sensor_token);
}

auto NuImages::get_sensor(const CalibratedSensor& calibrated_sensor) const
    -> const Sensor&
{
  return sensor_table.at(calibrated_sensor.sensor_token);
}

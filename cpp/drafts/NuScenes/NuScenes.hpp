#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/filesystem.hpp>

#include <Eigen/Eigen>

#include <nlohmann/json.hpp>


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

  auto load_json(const std::string& table_name) -> nlohmann::json
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

  auto load_sample_table() -> void
  {
    const auto sample_json = load_json("sample");
    for (const auto& j : sample_json)
      sample_table[j["token"]] = {
          .prev = j["prev"],
          .next = j["next"],
          .scene_token = j["scene_token"],
          .timestamp = j["timestamp"].get<std::uint64_t>()  //
      };
  }

  auto load_sample_data_table() -> void
  {
    const auto sample_data_json = load_json("sample_data");
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

  auto load_sample_annotation_table() -> void
  {
    const auto sample_annotation_json = load_json("sample_annotation");
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

  auto load_category_table() -> void
  {
    const auto category_json = load_json("category");

    category_table.reserve(category_json.size());
    for (const auto& j : category_json)
      category_table[j["token"]] = {j["name"], j["description"], j["index"]};

    for (const auto& [token, category] : category_table)
      std::cout << token << " " << category.index << " " << category.name
                << std::endl;
  }

  auto load_ego_pose_table() -> void
  {
    const auto ego_pose_json = load_json("ego_pose");

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

  auto load_calibrated_sensor_table() -> void
  {
    const auto calibrated_sensor_json = load_json("calibrated_sensor");

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

  auto load_instance_table() -> void
  {
    const auto instance_json = load_json("instance");

    instance_table.reserve(instance_json.size());
    for (const auto& j : instance_json)
    {
      auto instance = Instance{};

      instance_table[j["token"]] = {
          .category_token = j["category_token"],
          .number_of_annotations = j["nbr_annotations"].get<int>(),
          .first_annotation_token = j["first_annotation_token"],
          .last_annotation_token = j["last_annotation_token"]};
    }
  }

  auto load_visibility_table() -> void
  {
    const auto visibility_json = load_json("visibility");

    visibility_table.reserve(visibility_json.size());
    for (const auto& j : visibility_json)
      visibility_table[j["token"]] = {
          .description = j["description"],  //
          .level = j["level"]               //
      };
  }

  auto load_attribute_table() -> void
  {
    const auto attribute_json = load_json("attribute");

    attribute_table.reserve(attribute_json.size());
    for (const auto& j : attribute_json)
      attribute_table[j["token"]] = {
          .name = j["name"],               //
          .description = j["description"]  //
      };
  }

  auto load_sensor_table() -> void
  {
    const auto sensor_json = load_json("sensor");

    sensor_table.reserve(sensor_json.size());
    for (const auto& j : sensor_json)
      sensor_table[j["token"]] = {
          .channel = j["channel"],      //
          .modality = j["modality"]  //
      };
  }

  template <typename T>
  auto filter_by_sample_token(const std::unordered_map<Token, T>& table,
                              const Token& value) const
  {
    auto rows = std::vector<T>{};
    std::for_each(table.cbegin(), table.cend(), [&](const auto& row) {
      if (row.second.sample_token == value)
        rows.emplace_back(row.second);
    });
    return rows;
  }

  // template <typename T>
  // auto filter_by_instance_token(const std::unordered_map<Token, T>& table,
  //                               const Token& value) const
  // {
  //   auto rows = std::vector<T>{};
  //   std::copy_if(
  //       table.cbegin(), table.cend(), std::back_inserter(rows),
  //       [&value](const T& row) { return row.instance_token == value; });
  //   return rows;
  // }
};

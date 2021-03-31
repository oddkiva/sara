#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
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
    int width;
    int height;

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

  std::unordered_map<Token, Sample> samples;
  std::unordered_map<Token, SampleData> sample_data;
  std::unordered_map<Token, SampleAnnotation> sample_annotations;
  std::unordered_map<Token, Category> categories;


  NuScenes(const std::string& version_,   //
           const std::string& dataroot_,  //
           bool verbose_,                 //
           float map_resolution = 0.1)
    : version{version_}
    , dataroot{dataroot_}
    , verbose{verbose_}
  {
    load_samples();
    load_sample_annotations();
    // load_categories();
  }

private:
  auto load_table(const std::string& table_name) -> nlohmann::json
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

  auto load_samples() -> void
  {
    const auto sample_table = load_table("sample");
    for (const auto& j : sample_table)
      samples[j["token"]] = {
          .prev = j["prev"],
          .next = j["next"],
          .scene_token = j["scene_token"],
          .timestamp = j["timestamp"].get<std::uint64_t>()  //
      };
  }

  auto load_sample_data() -> void
  {
    const auto sample_data_table = load_table("sample_data");
    for (const auto& j : sample_data_table)
      sample_data[j["token"]] = {
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
          .filename = j["filename"]
      };
  }

  auto load_sample_annotations() -> void
  {
    const auto sample_annotation_table = load_table("sample_annotation");
    for (const auto& j : sample_annotation_table)
    {
      auto sample_annotation = SampleAnnotation{};
      sample_annotation.prev = j["prev"];
      sample_annotation.next = j["next"];
      sample_annotation.sample_token = j["sample_token"];
      sample_annotation.visibility_token = std::stoi(j["visibility_token"].get<std::string>());
      sample_annotation.instance_token = j["instance_token"];
      // for (const auto& jt: j["attributes_tokens"])
      //   sample_annotation.attributes_tokens.push_back(jt.get<std::string>());

      for (auto i = 0; i < 3; ++i)
        sample_annotation.translation(i) = j["translation"][i].get<float>();
      std::cout << sample_annotation.translation.transpose() << std::endl;

      for (auto i = 0; i < 3; ++i)
        sample_annotation.size(i) = j["size"][i].get<float>();

      for (auto i = 0; i < 4; ++i)
        sample_annotation.rotation = Eigen::Quaternionf{
            j["rotation"][0].get<float>(), j["rotation"][1].get<float>(),
            j["rotation"][2].get<float>(), j["rotation"][3].get<float>()};


      sample_annotation.num_lidar_pts = j["num_lidar_pts"].get<int>();
      sample_annotation.num_radar_pts = j["num_radar_pts"].get<int>();

      sample_annotations[j["token"]] = sample_annotation;
    }
  }

  auto load_categories() -> void
  {
    const auto category_table = load_table("category");

    categories.reserve(category_table.size());
    for (const auto& j : category_table)
      categories[j["token"]] = {j["name"], j["description"], j["index"]};

    for (const auto& [token, category] : categories)
      std::cout << token << " " << category.index << " " << category.name
                << std::endl;
  }
};

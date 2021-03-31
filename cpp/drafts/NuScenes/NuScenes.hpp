#pragma once

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include <nlohmann/json.hpp>


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

  struct Category
  {
    std::string token;
    std::string name;
    std::string description;
    int index;
  };

  std::vector<Category> categories;
  nlohmann::json category;
  nlohmann::json attribute;
  nlohmann::json visibility;
  nlohmann::json instance;
  nlohmann::json sensor;
  nlohmann::json calibrated_sensor;
  nlohmann::json ego_pose;
  nlohmann::json log;
  nlohmann::json scene;
  nlohmann::json sample;
  nlohmann::json sample_data;
  nlohmann::json sample_annotation;
  nlohmann::json map;


  NuScenes(const std::string& version_,   //
           const std::string& dataroot_,  //
           bool verbose_,                 //
           float map_resolution = 0.1)
    : version{version_}
    , dataroot{dataroot_}
    , verbose{verbose_}
  {
    category = load_table("category");
    attribute = load_table("attribute");
    visibility = load_table("visibility");
    instance = load_table("instance");
    sensor = load_table("sensor");
    calibrated_sensor = load_table("calibrated_sensor");
    ego_pose = load_table("ego_pose");
    log = load_table("log");
    scene = load_table("scene");
    sample = load_table("sample");
    sample_data = load_table("sample_data");
    sample_annotation = load_table("sample_annotation");
    map = load_table("map");
  }

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

    if (table_name == table_names[0])
    {
      categories.reserve(table_json.size());
      for (const auto& j : table_json)
        categories.push_back({
            j["token"], j["name"], j["description"], j["index"]  //
        });

      for (const auto& category : categories)
        std::cout << category.index << " " << category.name << std::endl;
    }

    return table_json;
  }
};

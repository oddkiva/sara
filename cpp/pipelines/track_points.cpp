// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2019 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/Sara/FeatureDetectors.hpp>
#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry/BundleAdjustmentProblem.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/SfM/BuildingBlocks.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace DO::Sara;


auto track_points(const std::string& dirpath, const std::string& h5_filepath,
                  bool /* overwrite */, bool /* debug */)
{
  // Create a backup.
  if (!fs::exists(h5_filepath + ".bak"))
    cp(h5_filepath, h5_filepath + ".bak");

  SARA_DEBUG << "Opening file " << h5_filepath << "..." << std::endl;
  auto h5_file = H5File{h5_filepath, H5F_ACC_RDWR};

  auto view_attributes = ViewAttributes{};

  // Load images (necessary if we want to extract the colors).
  SARA_DEBUG << "Listing images from:\n\t" << dirpath << std::endl;
  view_attributes.list_images(dirpath);

  // Load keypoints.
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath
             << std::endl;
  view_attributes.read_keypoints(h5_file);


  // Initialize the epipolar graph.
  const auto num_vertices = int(view_attributes.image_paths.size());
  SARA_CHECK(num_vertices);

  auto edge_attributes = EpipolarEdgeAttributes{};
  SARA_DEBUG << "Initializing the epipolar edges..." << std::endl;
  edge_attributes.initialize_edges(num_vertices);

  SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath
             << std::endl;
  edge_attributes.read_matches(h5_file, view_attributes);

  SARA_DEBUG << "Reading the essential matrices..." << std::endl;
  edge_attributes.resize_essential_edge_list();
  edge_attributes.read_essential_matrices(view_attributes, h5_file);

  SARA_DEBUG << "Reading the two-view geometries..." << std::endl;
  edge_attributes.read_two_view_geometries(view_attributes, h5_file);


  // Populate the feature tracks.
  const auto [feature_graph, components] =
      populate_feature_tracks(view_attributes, edge_attributes);

  // Save the graph of features in HDF5 format.
  write_feature_graph(feature_graph, h5_file, "feature_graph");

  // Postprocess the feature tracks, i.e.:
  // - Keep feature tracks of size 2 at least.
  // - A feature tracks should contain only one 2D feature point in each image,
  //   if it has more than one feature point in one image, we keep the feature
  //   point with the strongest feature detection response.
  const auto feature_tracks =
      filter_feature_tracks(feature_graph, components, view_attributes);
}


auto __main(int argc, char** argv) -> int
{
  try
  {
    po::options_description desc{"Estimate essential matrices"};
    desc.add_options()                                                 //
        ("help, h", "Help screen")                                     //
        ("dirpath", po::value<std::string>(), "Image directory path")  //
        ("out_h5_file", po::value<std::string>(), "Output HDF5 file")  //
        ("debug", "Inspect visually the epipolar geometry")            //
        ("overwrite", "Overwrite triangulation")                       //
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
      std::cout << desc << std::endl;
      return 0;
    }

    if (!vm.count("dirpath"))
    {
      std::cout << "[--dirpath]: missing image directory path" << std::endl;
      return 0;
    }

    if (!vm.count("out_h5_file"))
    {
      std::cout << desc << std::endl;
      std::cout << "[--out_h5_file]: missing output H5 file path" << std::endl;
      return 0;
    }

    const auto dirpath = vm["dirpath"].as<std::string>();
    const auto h5_filepath = vm["out_h5_file"].as<std::string>();
    const auto overwrite = vm.count("overwrite");
    const auto debug = vm.count("debug");

    track_points(dirpath, h5_filepath, overwrite, debug);

    return 0;
  }
  catch (const po::error& e)
  {
    std::cerr << e.what() << "\n";
    return 1;
  }
}


auto main(int argc, char** argv) -> int
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

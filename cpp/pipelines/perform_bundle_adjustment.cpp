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

#include <DO/Sara/FileSystem.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/MultiViewGeometry/FeatureGraph.hpp>
#include <DO/Sara/MultiViewGeometry/PoseGraph.hpp>
#include <DO/Sara/SfM/BuildingBlocks.hpp>
#include <DO/Sara/SfM/Detectors/SIFT.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace DO::Sara;


auto perform_bundle_adjustment(const std::string& dirpath,
                               const std::string& h5_filepath, bool overwrite,
                               bool debug)
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
  SARA_DEBUG << "Reading keypoints from HDF5 file:\n\t" << h5_filepath << std::endl;
  view_attributes.read_keypoints(h5_file);


  // Initialize the epipolar graph.
  const auto num_vertices = int(view_attributes.image_paths.size());
  SARA_CHECK(num_vertices);

  auto edge_attributes = EpipolarEdgeAttributes{};
  SARA_DEBUG << "Initializing the epipolar edges..." << std::endl;
  edge_attributes.initialize_edges(num_vertices);

  SARA_DEBUG << "Reading matches from HDF5 file:\n\t" << h5_filepath << std::endl;
  edge_attributes.read_matches(h5_file, view_attributes);

  SARA_DEBUG << "Reading the essential matrices..." << std::endl;
  edge_attributes.resize_essential_edge_list();
  edge_attributes.read_essential_matrices(view_attributes, h5_file);

  SARA_DEBUG << "Reading the two-view geometries..." << std::endl;
  edge_attributes.read_two_view_geometries(view_attributes, h5_file);

  // Convenient references.
  const auto& edge_ids = edge_attributes.edge_ids;
  const auto& edges = edge_attributes.edges;
  const auto& E_inliers = edge_attributes.E_inliers;

  // Populate the vertices.
  const auto image_ids = range(num_vertices);
  SARA_CHECK(image_ids.row_vector());
  auto pose_graph = PoseGraph(num_vertices);

  // Populate the edges.
  std::for_each(std::begin(edge_ids), std::end(edge_ids), [&](const auto& ij) {
    const auto& eij = edges[ij];
    const auto i = eij.first;
    const auto j = eij.second;
    const auto& inliers_ij = E_inliers[ij];
    const auto& cheirality_ij =
        edge_attributes.two_view_geometries[ij].cheirality;

    std::cout << std::endl;
    SARA_DEBUG << "Processing image pair " << i << " " << j << std::endl;

    SARA_DEBUG << "Checking if there are inliers..." << std::endl;
    SARA_CHECK(cheirality_ij.count());
    SARA_CHECK(inliers_ij.flat_array().count());
    if (inliers_ij.flat_array().count() == 0)
    {
      SARA_DEBUG << "NO INLIERS: SKIPPING..." << std::endl;
      return;
    }

    auto [e, b] = boost::add_edge(i, j, pose_graph);
    if (!b)
      throw std::runtime_error{
          "Error: failed to add edge: edge already exists!"};

    SARA_DEBUG << "Calculating cheiral inliers..." << std::endl;
    SARA_CHECK(cheirality_ij.size());
    SARA_CHECK(inliers_ij.size());
    if (cheirality_ij.size() != inliers_ij.size())
        throw std::runtime_error{"cheirality_ij.size() != inliers_ij.size()"};

    const Array<bool, 1, Dynamic> cheiral_inliers =
        inliers_ij.row_vector().array() && cheirality_ij;

    const auto num_cheiral_inliers = cheiral_inliers.count();
    SARA_CHECK(num_cheiral_inliers);

    pose_graph[i].weight += num_cheiral_inliers;
    pose_graph[j].weight += num_cheiral_inliers;
    pose_graph[e].id = ij;
    pose_graph[e].weight = num_cheiral_inliers;
  });

  SARA_DEBUG << "Inspecting vertex weights..." << std::endl;
  for (auto [v, v_end] = boost::vertices(pose_graph); v != v_end; ++v)
    SARA_DEBUG << format("weight[%d] = %f", *v, pose_graph[*v].weight)
               << std::endl;

  SARA_DEBUG << "Inspecting edge weights..." << std::endl;
  for (auto [e, e_end] = boost::edges(pose_graph); e != e_end; ++e)
    SARA_DEBUG << format("weight[%d] = %f", pose_graph[*e].id,
                         pose_graph[*e].weight)
               << std::endl;

  write_pose_graph(pose_graph, h5_file, "pose_graph");

  // TODO: Perform incremental bundle adjustment using a Dijkstra growing scheme.
  //
  // Let's just choose a heuristics for the incremental bundle adjustment even
  // if Bundler does not do like this.
  view_attributes.cameras.resize(num_vertices);

  // Reload the point tracks.
  const auto feature_graph = read_feature_graph(h5_file, "feature_graph");

  // TODO: collect the 3D points visible over the set of images.
  // 1. Initialize the absolute camera poses from the relative camera poses.
  // 2. Recalculate the 3D points in the world coordinate frame.

  // TODO: readapt the Ceres sample code for bundle adjustment.
#ifdef CERES_BUNDLE_ADJUSTMENT
  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i)
  {
    auto cost_fn =
        SnavelyReprojectionError::Create(bal_problem.observations()[2 * i + 0],
                                         bal_problem.observations()[2 * i + 1]);

    problem.AddResidualBlock(cost_function,
                             nullptr /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }

  auto options = ceres::Solver::Options{};
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  auto summary = ceres::Solver::Summary{};

  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
#endif

  // TODO: save the point cloud.

  // TODO: display the cameras on OpenGL
}


int __main(int argc, char **argv)
{
  try
  {
    po::options_description desc{"Perform bundle adjustment"};
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

    perform_bundle_adjustment(dirpath, h5_filepath, overwrite, debug);

    return 0;
  }
  catch (const po::error& e)
  {
    std::cerr << e.what() << "\n";
    return 1;
  }
}


int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

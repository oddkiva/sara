#include <drafts/NuScenes/NuScenes.hpp>


int main()
{
  using namespace std::string_literals;

  const auto nuscenes_version = "v1.0-mini"s;
  const auto nuscenes_root_path = "/home/david/Downloads/nuscenes"s;

  const auto nuscenes = NuScenes{nuscenes_version, nuscenes_root_path, true};

  // A sample is referenced by its token.
  const auto& sample_token = nuscenes.sample_table.begin()->first;
  std::cout << "sample_token = " << sample_token << std::endl;


  // Retrieve the list of sample data, i.e.:
  // - images
  // - lidar point cloud
  // - radar velocities
  const auto sample_data = nuscenes.filter_by_sample_token(  //
      nuscenes.sample_data_table,                            //
      sample_token                                           //
  );
  std::cout << "Number of sample data = " << sample_data.size() << std::endl;
  std::cout << "List of sample data files" << std::endl;
  for (const auto& data : sample_data)
    std::cout << data.filename << std::endl;
  std::cout << std::endl;


  // Number of annotated objects:
  const auto sample_annotations = nuscenes.filter_by_sample_token(  //
      nuscenes.sample_annotation_table,                                  //
      sample_token                                                  //
  );
  std::cout << "Number of annotated objects = " << sample_annotations.size()
            << std::endl;
  std::cout << "List of annotations" << std::endl;
  for (const auto& data : sample_annotations)
  {
    std::cout << "s = " << data.size.transpose() << "  "
              << "t = " << data.translation.transpose() << "  "
              << "R = " << data.rotation.coeffs().transpose() << std::endl;
  }
  std::cout << std::endl;


  // Check that we can query the ego pose data from each sample data.
  for (const auto& datum: sample_data)
  {
    const auto& ego_pose = nuscenes.ego_pose_table.at(datum.ego_pose_token);
    std::cout << "datum.timestamp = " << datum.timestamp << std::endl;
    std::cout << "ego_pose.timestamp = " << ego_pose.timestamp << std::endl;
    std::cout << "ego_pose.rotation = " << ego_pose.rotation.toRotationMatrix()
              << std::endl;
  }

  return 0;
}

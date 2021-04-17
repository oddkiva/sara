#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <drafts/NuScenes/NuScenes.hpp>


namespace sara = DO::Sara;


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

  const auto nuscenes_version = "v1.0-mini"s;
  const auto nuscenes_root_path = "/home/david/Downloads/nuscenes"s;

  const auto nuscenes = NuScenes{nuscenes_version, nuscenes_root_path, true};

  // A sample is indexed by its token.
  //
  // It also has a unique timestamp.
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

  // We just want to pull one image.
  const auto& image_metadata = *std::find_if(             //
      sample_data.cbegin(), sample_data.cend(),           //
      [](const auto& datum) { return datum.is_image(); }  //
  );

  const auto image = sara::imread<sara::Rgb8>(  //
      nuscenes.get_data_path(image_metadata)    //
  );

  // Display the image.
  sara::create_window(image.sizes());
  sara::set_antialiasing();
  sara::display(image);

  // Get the ego pose, which the car vehical position and orientation with
  // respect to the global coordinate system.
  const auto& ego_pose = nuscenes.get_ego_pose(image_metadata);
  // In the world reference, the car is at the following coordinates:
  const auto& tw = ego_pose.translation;
  // In the world reference, the car orientation is quantified by a quaternion:
  // We can transform into a rotation matrix.
  const auto& Rw = ego_pose.rotation.toRotationMatrix();

  std::cout << "Comparing the timestamps" << std::endl;
  std::cout << nuscenes.sample_table.at(sample_token).timestamp << std::endl;
  std::cout << image_metadata.timestamp << std::endl;
  std::cout << ego_pose.timestamp << std::endl;

  // Camera poses with respect to the reference local vehicle coordinate system.
  //
  // And I am pretty sure the IMU coordinate system is used as the reference
  // local vehicle coordinate system.
  const auto& calibrated_sensor = nuscenes.get_calibrated_sensor(image_metadata);
  // Get the camera pose w.r.t. the local coordinate system.
  const auto& tc = calibrated_sensor.translation;
  const auto& Rc = calibrated_sensor.rotation.toRotationMatrix();
  // Get the calibration matrix.
  const auto& K = calibrated_sensor.calibration_matrix;
  // Upon visual inspection, this sign of each camera axis coordinate are
  // consistent with the projection of each axis onto the IMU axes.
  std::cout << "R(camera/local) =\n" << Rc << std::endl;

  // Now get the sensor type that acquired the image.
  const auto& sensor = nuscenes.get_sensor(calibrated_sensor);
  sara::draw_text(10, 20, sensor.modality + ": " + sensor.channel, sara::White8,
                  16, 0, false, true);

  // Let us now pull the list of annotated objects:
  std::cout << "Listing the annotated objects..." << std::endl;
  const auto sample_annotations = nuscenes.filter_by_sample_token(  //
      nuscenes.sample_annotation_table,                             //
      sample_token                                                  //
  );

  for (const auto& annotation: sample_annotations)
  {
    // For now, just consider the center of the 3D box.
    //
    // Then we will proceed with all the vertices of the full box after that.
    const auto& x_global = annotation.translation;

    // Get the coordinates in the reference local coordinate system.
    const Eigen::Vector3f& x_local = Rw.transpose() * (x_global - tw);

    // Get the camera coordinates from the reference local coordinate system.
    const Eigen::Vector3f& x_camera = Rc.transpose() * (x_local - tc);

    // Project onto the image plane.
    const Eigen::Vector2f x_image = (K.value() * x_camera).hnormalized();

    // Inspect the label of the annotation to see if they match with what we see
    // in the image.
    if (x_image.x() < 0 || x_image.x() > image.width() ||  //
        x_image.y() < 0 || x_image.y() > image.height())
      continue;

    const auto& instance =
        nuscenes.instance_table.at(annotation.instance_token);
    const auto& category = nuscenes.category_table.at(instance.category_token);
    std::cout << category.name << " " << x_camera.transpose() << std::endl;

    sara::fill_circle(x_image, 3.f, sara::Red8);

    // There are some errors to me and missing annotations.
  }

  sara::get_key();

  return 0;
}

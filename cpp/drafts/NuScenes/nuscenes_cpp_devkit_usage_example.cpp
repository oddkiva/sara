#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <drafts/NuScenes/NuScenes.hpp>


namespace sara = DO::Sara;

struct Box
{
  //! @brief Height, width, length.
  Eigen::Vector3f size;

  //! @brief the rotation matrix.
  Eigen::Matrix3f R;

  //! @brief the translation vector.
  //!
  //! This is also the center of the bounding box in the world reference
  //! coordinate system.
  Eigen::Vector3f t;

  auto vertices() const
  {
    // width -> x-axis
    const auto& w = size(0);
    // length -> y-axis
    const auto& l = size(1);
    // height -> z-axis
    const auto& h = size(2);

    // The coordinates follow the automotive axis convention.
    auto vertices = Eigen::Matrix<float, 3, 8>{};
    vertices <<
      // Back face   | Front face
      +1, +1, +1, +1, -1, -1, -1, -1,
      +1, -1, -1, +1, +1, -1, -1, +1,
      +1, +1, -1, -1, +1, +1, -1, -1;

    // The first 4 vertices are the back face, enumerated in the following
    // order:
    // 1. Top-left
    // 2. Top-right
    // 3. Bottom-right
    // 4. Bottom-left
    //
    // The last 4 vertices are the front face, also  enumerated in the following
    // order:
    // 1. Top-left
    // 2. Top-right
    // 3. Bottom-right
    // 4. Bottom-left

    // Multiply the coordinates with the appropriate size.
    auto x = vertices.row(0);
    auto y = vertices.row(1);
    auto z = vertices.row(2);

    x *= l * 0.5f;
    y *= w * 0.5f;
    z *= h * 0.5f;

    // Rotation (yaw essentially)
    vertices = R * vertices;
    // Then translate in the world reference coordinate system.
    vertices.colwise() += t;

    return vertices;
  }

  auto drawBBox3D(const NuScenes::EgoPose& ego_pose,
                  const NuScenes::CalibratedSensor& calibrated_sensor) const
  {
    Eigen::Matrix<float, 4, 8> X = vertices().colwise().homogeneous();

    Eigen::Matrix4f world_to_local_transform = Eigen::Matrix4f::Identity();
    {
      const Eigen::Matrix3f Rwt = ego_pose
                                      .rotation            //
                                      .toRotationMatrix()  //
                                      .transpose();        //
      const auto& tw = ego_pose.translation;

      world_to_local_transform.topLeftCorner(3, 3) = Rwt;
      world_to_local_transform.block<3, 1>(0, 3) = -Rwt * tw;
    }

    Eigen::Matrix4f local_to_camera_transform = Eigen::Matrix4f::Identity();
    {
      const Eigen::Matrix3f Rct = calibrated_sensor        //
                                      .rotation            //
                                      .toRotationMatrix()  //
                                      .transpose();
      const auto& tc = calibrated_sensor.translation;

      local_to_camera_transform.topLeftCorner(3, 3) = Rct;
      local_to_camera_transform.block<3, 1>(0, 3) = -Rct * tc;
    }

    const Eigen::Matrix4f world_to_camera_transform =
        local_to_camera_transform * world_to_local_transform;

    X = world_to_camera_transform * X;

    // Project to camera.
    const auto& K = calibrated_sensor.calibration_matrix.value();
    const Eigen::Matrix<float, 3, 8> x = K * X.colwise().hnormalized();

    // Draw the vertices.
    for (auto v = 0; v < 8; ++v)
    {
      const Eigen::Vector2f uv = x.col(v).head(2);
      sara::fill_circle(uv, 3.f, sara::Magenta8);
    }

    // Draw the edges of the back-face.
    for (auto v = 0; v < 4; ++v)
    {
      const Eigen::Vector2f a = x.col(v).head(2);
      const Eigen::Vector2f b = x.col((v + 1) % 4).head(2);
      sara::draw_line(a, b, sara::Red8, 1);
    }

    // Draw the edges of the front-face.
    for (auto v = 0; v < 4; ++v)
    {
      const Eigen::Vector2f a = x.col(4 + v).head(2);
      const Eigen::Vector2f b = x.col(4 + (v + 1) % 4).head(2);
      sara::draw_line(a, b, sara::Red8, 1);
    }

    // Draw the edges of the left and right side faces.
    for (auto v = 0; v < 4; ++v)
    {
      const Eigen::Vector2f a = x.col(0 + v).head(2);
      const Eigen::Vector2f b = x.col(4 + v).head(2);
      sara::draw_line(a, b, sara::Red8, 1);
    }
  }
};


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
  const auto& calibrated_sensor =
      nuscenes.get_calibrated_sensor(image_metadata);
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

  for (const auto& annotation : sample_annotations)
  {
    // For now, just consider the center of the 3D box.
    //
    // Then we will proceed with all the vertices of the full box after that.
    const auto& x_global = annotation.translation;

    // Get the coordinates in the reference local coordinate system.
    const Eigen::Vector3f& x_local = Rw.transpose() * (x_global - tw);

    // Get the camera coordinates from the reference local coordinate system.
    const Eigen::Vector3f& x_camera = Rc.transpose() * (x_local - tc);

    // Check the cheirality: z > 0!
    if (x_camera.z() < 0)
      continue;

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
    std::cout << "size = " << annotation.size.transpose() << std::endl;


    sara::fill_circle(x_image, 3.f, sara::Red8);

    // There are some errors to me and missing annotations.
  }

  sara::get_key();

  return 0;
}

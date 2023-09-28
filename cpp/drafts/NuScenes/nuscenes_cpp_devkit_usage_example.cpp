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

//! @file

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include <drafts/NuScenes/NuScenes.hpp>


namespace sara = DO::Sara;


// The following also works.
// auto rigid_body_transform(const Eigen::Quaternionf& R, const Eigen::Vector3f&
// t)
//     -> Eigen::Affine3f
// {
//   return R * Eigen::Translation3f{t};
// }

// In the world reference, the car is:
// - at the following coordinates given by 'ego_pose.translation'
// - oriented by the quaternion 'ego_pose.rotation'
auto local_to_world_transform(const NuScenes::EgoPose& ego_pose)
{
  // return rigid_body_transform(ego_pose.rotation, ego_pose.translation)

  const Eigen::Matrix3f R = ego_pose
                                .rotation            //
                                .toRotationMatrix()  //
                                .transpose();        //
  const auto& t = ego_pose.translation;

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = R;
  T.block<3, 1>(0, 3) = t;

  return T;
}

auto world_to_local_transform(const NuScenes::EgoPose& ego_pose)
{
  // return rigid_body_transform(ego_pose.rotation, ego_pose.translation)
  //     .inverse();

  const Eigen::Matrix3f Rt = ego_pose
                                 .rotation            //
                                 .toRotationMatrix()  //
                                 .transpose();        //
  const auto& t = ego_pose.translation;

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.topLeftCorner(3, 3) = Rt;
  T.block<3, 1>(0, 3) = -Rt * t;

  return T;
}

auto local_to_sensor_transform(
    const NuScenes::CalibratedSensor& calibrated_sensor)
{
  const Eigen::Matrix3f Rt = calibrated_sensor        //
                                 .rotation            //
                                 .toRotationMatrix()  //
                                 .transpose();
  const auto& t = calibrated_sensor.translation;

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

  T.topLeftCorner(3, 3) = Rt;
  T.block<3, 1>(0, 3) = -Rt * t;

  return T;
}


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

  std::string category_name;

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
        +1, +1, +1, +1, -1, -1, -1, -1,  //
        +1, -1, -1, +1, +1, -1, -1, +1,  //
        +1, +1, -1, -1, +1, +1, -1, -1;  //

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

  auto draw(const NuScenes::EgoPose& ego_pose,
            const NuScenes::CalibratedSensor& calibrated_sensor) const
  {
    const Eigen::Matrix4f T = local_to_sensor_transform(calibrated_sensor) *
                              world_to_local_transform(ego_pose);

    const Eigen::Matrix<float, 4, 8> X = T * vertices().colwise().homogeneous();

    // Project to camera.
    const auto& K = calibrated_sensor.calibration_matrix.value();
    const Eigen::Matrix<float, 2, 8> x = (K * X.colwise().hnormalized())  //
                                             .colwise()
                                             .hnormalized();

    // Draw the vertices.
    for (auto v = 0; v < 8; ++v)
    {
      const Eigen::Vector2f uv = x.col(v);
      sara::fill_circle(uv, 3.f, sara::Magenta8);
    }

    // Draw the edges of the back-face.
    for (auto v = 0; v < 4; ++v)
    {
      const Eigen::Vector2f a = x.col(v);
      const Eigen::Vector2f b = x.col((v + 1) % 4);
      sara::draw_line(a, b, sara::Red8, 1);
    }

    // Draw the edges of the front-face.
    for (auto v = 0; v < 4; ++v)
    {
      const Eigen::Vector2f a = x.col(4 + v);
      const Eigen::Vector2f b = x.col(4 + (v + 1) % 4);
      sara::draw_line(a, b, sara::Red8, 1);
    }

    // Draw the edges of the left and right side faces.
    for (auto v = 0; v < 4; ++v)
    {
      const Eigen::Vector2f a = x.col(0 + v);
      const Eigen::Vector2f b = x.col(4 + v);
      sara::draw_line(a, b, sara::Red8, 1);
    }

    const Eigen::Vector2i tl = x.rowwise().minCoeff().cast<int>();
    sara::draw_text(tl.x() - 20, tl.y() - 10, category_name, sara::White8, 16,
                    0, false, true, false, 2);
  }
};


GRAPHICS_MAIN()
{
  using namespace std::string_literals;

  const auto nuscenes_version = "v1.0-mini"s;
  const auto nuscenes_root_path = "/home/david/Downloads/nuscenes"s;

  const auto nuscenes = NuScenes{nuscenes_version, nuscenes_root_path, true};

  for (const auto& [sample_token, sample] : nuscenes.sample_table)
  {
    // A sample is indexed by its token.
    //
    // It also has a unique timestamp.
    std::cout << "sample_token = " << sample_token << std::endl;

    // Retrieve the list of sample data, i.e.:
    // - images
    // - lidar point cloud
    // - radar velocities
    const auto sample_data = nuscenes.filter_by_sample_token(  //
        nuscenes.sample_data_table,                            //
        sample_token                                           //
    );

    // Display all the images.
    for (const auto& sample_datum : sample_data)
    {
      if (!sample_datum.is_image())
        continue;

      if (!sample_datum.is_key_frame)
        continue;

      const auto image =
          sara::imread<sara::Rgb8>(nuscenes.get_data_path(sample_datum));

      // Get the ego pose, which the car vehical position and orientation with
      // respect to the global coordinate system.
      const auto& ego_pose = nuscenes.get_ego_pose(sample_datum);

      // Camera poses with respect to the reference local vehicle coordinate
      // system.
      //
      // And I am pretty sure the IMU coordinate system is used as the reference
      // local vehicle coordinate system.
      const auto& calibrated_sensor =
          nuscenes.get_calibrated_sensor(sample_datum);
      // Get the calibration matrix.
      const auto& K = calibrated_sensor.calibration_matrix.value();

      // Transformation to go from the world coordinate system to the sensor
      // coordinate system.
      const Eigen::Matrix4f T = local_to_sensor_transform(calibrated_sensor) *
                                world_to_local_transform(ego_pose);

      // Now get the sensor type that acquired the image.
      const auto& sensor = nuscenes.get_sensor(calibrated_sensor);
      sara::draw_text(10, 20, sensor.modality + ": " + sensor.channel,
                      sara::White8, 16, 0, false, true);

      // Let us now pull the list of annotated objects:
      const auto sample_annotations = nuscenes.filter_by_sample_token(  //
          nuscenes.sample_annotation_table,                             //
          sample_token                                                  //
      );

      // Display the image.
      if (!sara::active_window())
      {
        sara::create_window(image.sizes());
        sara::set_antialiasing();
      }
      else if (sara::get_sizes(sara::active_window()) != image.sizes())
        sara::resize_window(image.sizes());
      sara::display(image);

      sara::draw_text(20, 20, sensor.channel, sara::White8, 16, 0, false, true,
                      false, 2);

      // Draw the visible bounding boxes.
      for (const auto& annotation : sample_annotations)
      {
        // For now, just consider the center of the 3D box.
        //
        // Then we will proceed with all the vertices of the full box after
        // that.
        const auto& c_global = annotation.translation;

        // Get the camera coordinates from the reference local coordinate
        // system.
        const Eigen::Vector3f& c_camera =
            (T * c_global.homogeneous()).hnormalized();

        // Check the cheirality: z > 0!
        if (c_camera.z() < 0)
          continue;

        // Project onto the image plane.
        const Eigen::Vector2f c_image = (K * c_camera).hnormalized();

        // Inspect the label of the annotation to see if they match with what we
        // see in the image.
        if (c_image.x() < 0 || c_image.x() > image.width() ||  //
            c_image.y() < 0 || c_image.y() > image.height())
          continue;

        const auto& instance =
            nuscenes.instance_table.at(annotation.instance_token);
        const auto& category =
            nuscenes.category_table.at(instance.category_token);

        const auto box = Box{annotation.size,
                             annotation.rotation.toRotationMatrix(),
                             annotation.translation,
                             category.name};

        box.draw(ego_pose, calibrated_sensor);
      }

      sara::get_key();
    }
  }

  return 0;
}

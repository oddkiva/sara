#pragma once

#include <vector>

#include <DO/Sara/Core.hpp>

#include "utilities.hpp"
#include "geometric_objects.hpp"


using std::vector;
using namespace DO::Sara;


class Scene
{
  Vector3d _camera_pos;
  Vector3d _light_pos;
  std::vector<GeometricObject *> _objects;

public:
  Scene(const Vector3d& camera_pos, const Vector3d& light_pos);

  ~Scene();

  void add_object(GeometricObject *object)
  {
    _objects.push_back(object);
  }

  void generate_scene(Image<Rgb64f>& image, Image<double>& z_buffer) const;
};

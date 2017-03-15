#include "raytracer.hpp"


using namespace std;
using namespace DO;


Scene::Scene(const Vector3d& cam_pos, const Vector3d& light_pos)
  : _camera_pos(cam_pos)
  , _light_pos(light_pos)
{
}

Scene::~Scene()
{
  _objects.clear();
}

void Scene::generate_scene(Image<Rgb64f>& I, Image<double>& z_buffer) const
{
  if (I.width() != w || I.height() != h)
    I = Image<Rgb64f>(w, h);

  if (z_buffer.width() != w || z_buffer.height() != h)
    z_buffer = Image<double>(w, h);

  I.flat_array().fill(Black64f);
  z_buffer.flat_array().fill(std::numeric_limits<double>::max());

  for (int y = -h / 2; y < h / 2; ++y)
  {
    for (int x = -w / 2; x < w / 2; ++x)
    {
      const auto p = Point3d{double(x), double(y), f};
      auto ray = Vector3d{p - _camera_pos};

      auto impact = Point3d{infinite_point};
      auto reflection = Black64f;

      for (auto o = _objects.begin(); o != _objects.end(); ++o)
      {
        if ((*o)->hit(impact, reflection, _camera_pos, ray, _light_pos) &&
          z_buffer(x + w / 2, y + h / 2) > impact.z())
        {
          z_buffer(x + w / 2, y + h / 2) = impact.z();
          I(x + w / 2, y + h / 2) = reflection;
        }
      }
    }
  }

  // Post-process the z_buffer for convenient display check
  for (int y = -h / 2; y < h / 2; ++y)
    for (int x = -w / 2; x < w / 2; ++x)
      if (z_buffer(x + w / 2, y + h / 2) == std::numeric_limits<double>::max())
        z_buffer(x + w / 2, y + h / 2) = 0;
}

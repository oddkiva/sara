#include <iostream>

#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>

#include "geometric_objects.hpp"
#include "raytracer.hpp"
#include "utilities.hpp"


using namespace std;
using namespace DO::Sara;


void create_scene_1(Scene& scene)
{
  const auto n1 = Vector3d(1, 0, 0);
  const auto p1 = Point3d(500, 0, 0);

  const auto n2 = Vector3d(0, 1, 0);
  const auto p2 = Point3d(0, 500, 0);

  const auto n3 = Vector3d(0, 0, 1);
  const auto p3 = Point3d(0, 0, 1000);

  scene.add_object(new HalfSpace(n1, p1, Red64f));
  scene.add_object(new HalfSpace(n2, p2, Green64f));
  scene.add_object(new HalfSpace(n3, p3, Blue64f));
}

void create_scene_2(Scene& scene)
{
  const auto c1 = Vector3d(50, -150, 500);
  const auto c2 = Vector3d(-200, 0, 650);
  const auto c3 = Vector3d(200, 200, 500);
  const auto c4 = Vector3d(-100, 200, 200);

  const auto n1 = Vector3d(1, 0, 0);
  const auto p1 = Point3d(500, 0, 0);

  const auto n2 = Vector3d(0, 1, 0);
  const auto p2 = Point3d(0, 500, 0);

  const auto n3 = Vector3d(0, 0, 1);
  const auto p3 = Point3d(0, 0, 1000);

  scene.add_object(new ::Sphere(c1, 200, Yellow64f));
  scene.add_object(new ::Sphere(c2, 100, Red64f));
  scene.add_object(new ::Sphere(c3, 75, Green64f));
  scene.add_object(new ::Sphere(c4, 150, Cyan64f));

  scene.add_object(new ::Cube(250));
}

void create_scene_3(Scene& scene)
{
  scene.add_object(new ::Cube(250));
}

GRAPHICS_MAIN()
{
  auto key = -1;
  auto x = 150., y = -400., z = -5000.;
  auto step = 200;

  const auto save_scene = false;
  const auto check_z_buffer = false;

  create_window(w, h);
  auto image = Image<Rgb64f>{};
  auto z_buffer = Image<double>{};
  do
  {
    // Default parameter.
    const auto cam_pos = Vector3d{x, y, z};
    const auto light_pos = Vector3d::Zero().eval();
    auto scene = Scene{cam_pos, light_pos};

    // Create scene by adding objects.
    create_scene_2(scene);

    // Generate sceneBuffer.
    scene.generate_scene(image, z_buffer);

    // Visualize resulted scene.
    display(image);

    if (check_z_buffer)
    {
      get_key();
      const auto m = z_buffer.flat_array().minCoeff();
      const auto M = z_buffer.flat_array().maxCoeff();
      cout << m << " " << M << endl;
      z_buffer.flat_array() = (z_buffer.flat_array() - m) / (M - m);
      display(z_buffer);
    }

    if (save_scene)
    {
      imwrite(image.convert<Rgb8>(), "scene.png");
      imwrite(z_buffer.convert<Rgb8>(), "z_buffer.png");
    }

    key = get_key();
    if (key == 'Z')
      y -= step;
    if (key == 'S')
      y += step;
    if (key == 'Q')
      x -= step;
    if (key == 'D')
      x += step;
    if (key == KEY_PAGEUP)
      z += step;
    if (key == KEY_PAGEDOWN)
      z -= step;
  } while (key != KEY_ESCAPE);

  return 0;
}

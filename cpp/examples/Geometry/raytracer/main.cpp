#include <iostream>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Geometry.hpp>
#include <DO/Sara/ImageIO.hpp>

#include "geometric_objects.hpp"
#include "raytracer.hpp"
#include "utilities.hpp"


using namespace std;
using namespace DO::Sara;


void create_scene_1(Scene& scene)
{
  Vector3d n1(1,0,0);
  Point3d p1(500,0,0);

  Vector3d n2(0,1,0);
  Point3d p2(0,500,0);

  Vector3d n3(0,0,1);
  Point3d p3(0,0,1000);

  scene.add_object(new HalfSpace(n1, p1, Red64f));
  scene.add_object(new HalfSpace(n2, p2, Green64f));
  scene.add_object(new HalfSpace(n3, p3, Blue64f));
}

void create_scene_2(Scene& scene)
{
  const Vector3d c1(50,-150,500);
  const Vector3d c2(-200,0,650);
  const Vector3d c3(200,200,500);
  const Vector3d c4(-100,200,200);

  Vector3d n1(1, 0, 0); Point3d p1(500, 0, 0);
  Vector3d n2(0, 1, 0); Point3d p2(0, 500, 0);
  Vector3d n3(0, 0, 1); Point3d p3(0, 0, 1000);

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
  int key = -1;
  double x = 150, y = -400, z = -5000;
  double step = 200;

  bool save_scene = false;
  bool check_z_buffer = false;

  create_window(w, h);
  Image<Rgb64f> image;
  Image<double> z_buffer;
  do
  {
    // Default parameter.
    Vector3d cam_pos(x,y,z);
    Vector3d light_pos(0,0,0);
    Scene scene(cam_pos, light_pos);

    // Create scene by adding objects.
    create_scene_2(scene);

    // Generate sceneBuffer.
    scene.generate_scene(image, z_buffer);

    // Visualize resulted scene.
    display(image);

    if (check_z_buffer)
    {
      get_key();
      double m = z_buffer.array().minCoeff();
      double M = z_buffer.array().maxCoeff();
      cout << m << " " << M << endl;
      z_buffer.array() = (z_buffer.array() - m) / (M-m);
      display(z_buffer);
    }

    if (save_scene)
    {
      imwrite(image.convert<Rgb8>(), "scene.png");
      imwrite(z_buffer.convert<Rgb8>(), "z_buffer.png");
    }

    key = get_key();
    if (key == 'Z')
      y-=step;
    if (key == 'S')
      y+=step;
    if (key == 'Q')
      x-=step;
    if (key == 'D')
      x+=step;
    if (key == KEY_PAGEUP)
      z+=step;
    if (key == KEY_PAGEDOWN)
      z-=step;
  } while (key != KEY_ESCAPE);

  return 0;
}

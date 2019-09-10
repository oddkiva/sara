#include <DO/Sara/Core.hpp>
#include <DO/Kalpana.hpp>

#include <QApplication>
#include <QSurfaceFormat>

#include <iostream>


void build_scene(DO::Kalpana::Scene& scene)
{
  using namespace std;
  using namespace DO::Kalpana;

  vector<Vector3f> points;
  for (int i = -10; i < 10; ++i)
    points.push_back(Vector3f(i, i, i));

  auto point_cloud = scene.scatter(points);

  // Put this in Canvas3D::initializeGL()
  auto vertex_shader = string{ R"(
    #version 330

    uniform mat4 proj_mat;
    uniform mat4 modelview_mat;

    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    layout(location = 2) in float size;

    out vec4 vert_color;

    void main(void)
    {
      gl_Position = proj_mat * modelview_mat * vec4(position, 1.0);
      gl_PointSize = abs(position[2]);
      vert_color = vec4(color, 1.0);
    })" };

  auto fragment_shader = string{ R"(
    #version 330

    in vec4 vert_color;
    out vec4 frag_color;

    void main(void)
    {
      frag_color = vert_color;
    })" };

  point_cloud->set_vertex_shader_source(vertex_shader);
  point_cloud->set_fragment_shader_source(fragment_shader);
}


int main(int argc, char **argv)
{
  using namespace std;
  using namespace DO::Kalpana;

  QSurfaceFormat format;
  format.setMajorVersion(3);
  format.setMinorVersion(3);
  format.setOption(QSurfaceFormat::DebugContext);
  format.setProfile(QSurfaceFormat::CoreProfile);
  QSurfaceFormat::setDefaultFormat(format);

  QApplication app{argc, argv};

  Scene scene{};
  build_scene(scene);

  Canvas3D ax{&scene};
  ax.setFormat(format);
  ax.resize(320, 240);
  ax.show();

  return app.exec();
}

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

//! @example

#include <QApplication>

#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/OpenGLWindow.hpp>
#include <DO/Sara/Graphics/DerivedQObjects/RotationSliders.hpp>

#include <regex>


using namespace std;
using namespace DO::Sara;


auto split(
    std::vector<std::string>& tokens, //
    const std::string& str,
    const std::string& delimiters = R"([\s,]+)")  // split on space and comma
{
  std::regex regex{delimiters.c_str()};
  std::sregex_token_iterator it{str.begin(), str.end(), regex, -1};

  tokens.clear();
  tokens.insert(tokens.end(), it, {});
}


namespace DO::Sara::v2 {

  class MeshReader
  {
  public:
    bool read_object_file(SimpleMesh<Vector3f, Face3>& mesh,
                          const std::string& fileName)
    {
      // Attempt to read file.
      std::ifstream file(fileName.c_str());
      if (!file)
      {
        std::cerr << "Error reading file!" << std::endl;
        return false;
      }
      // Clear the mesh data structure.
      mesh = SimpleMesh<Vector3f, Face3>{};
      // Fill the mesh data structure.
      auto line = std::string{};
      auto tokens = std::vector<std::string>{};
      auto face_tokens = std::vector<std::string>{};
      tokens.reserve(4);
      face_tokens.reserve(3);
      while (std::getline(file, line))
      {
        split(tokens, line, R"([\s]+)");
        const auto& type = tokens.front();
        if (type == "v")
        {
          auto v = Eigen::Vector3f{};
          for (auto i = 0; i < 3; ++i)
            v[i] = std::stod(tokens[i + 1]);
          mesh.vertices().emplace_back(v);
        }

        // Parse the more sophisticated form of face data.
        if (type == "f")
        {
          auto f = Face3{};
          for (auto i = 0; i < 3; ++i)
          {
            split(face_tokens, tokens[i + 1], R"([//]+)");
            f[i] = std::stoi(face_tokens.front()) - 1;
          }
          mesh.faces().push_back(f);
        }
      }

      // Read mesh successfully.
      compute_face_rings(mesh);
      compute_vertex_rings(mesh);
      compute_normals(mesh);
      return true;
    }

  private:
    template <typename Vector, typename Face>
    void compute_vertex_rings(const SimpleMesh<Vector, Face>& mesh)
    {
      vertex_rings_.resize(mesh.vertices().size());
      // Add neighboring vertices.
      for (size_t f = 0; f != mesh.faces().size(); ++f)
      {
        for (int v = 0; v < 3; ++v)
        {
          size_t v1 = mesh.face(f)(v);
          size_t v2 = mesh.face(f)((v + 1) % 3);
          size_t v3 = mesh.face(f)((v + 2) % 3);
          vertex_rings_[v1].push_back(v2);
          vertex_rings_[v1].push_back(v3);
        }
      }
      // Eliminate redundancies.
      for (size_t r = 0; r != vertex_rings_.size(); ++r)
      {
        std::vector<size_t>& vertexRing = vertex_rings_[r];
        std::sort(vertexRing.begin(), vertexRing.end());
        typename std::vector<size_t>::iterator it =
            std::unique(vertexRing.begin(), vertexRing.end());
        vertexRing.resize(it - vertexRing.begin());
      }
    }

    template <typename Vector, typename Face>
    void compute_face_rings(const SimpleMesh<Vector, Face>& mesh)
    {
      face_rings_.resize(mesh.vertices().size());
      for (size_t f = 0; f != mesh.faces().size(); ++f)
      {
        face_rings_[mesh.face(f)(0)].push_back(f);
        face_rings_[mesh.face(f)(1)].push_back(f);
        face_rings_[mesh.face(f)(2)].push_back(f);
      }
    }

    template <typename Vector, typename Face>
    Vector compute_vertex_normals(const SimpleMesh<Vector, Face>& mesh,
                                  size_t v)
    {
      Vector n(Vector::Zero());
      const std::vector<size_t>& faceRing = face_rings_[v];
      for (size_t t = 0; t < faceRing.size(); ++t)
        n += mesh.face_normal(faceRing[t]);
      n /= static_cast<float>(faceRing.size());
      n.normalize();
      return n;
    }

    template <typename Vector, typename Face>
    void compute_normals(SimpleMesh<Vector, Face>& mesh)
    {
      mesh.normals().resize(face_rings_.size());
      for (size_t v = 0; v != face_rings_.size(); ++v)
        mesh.normal(v) = compute_vertex_normals(mesh, v);
    }

  private:
    std::vector<std::vector<size_t>> vertex_rings_;
    std::vector<std::vector<size_t>> face_rings_;
  };

}  // namespace DO::Sara::v2


// #define USE_SARA_API
#ifdef USE_SARA_API
// Hacky... whatever.
RotationSliders* some_slider = nullptr;

int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);

  auto rotation_sliders = RotationSliders{};
  some_slider = &rotation_sliders;

  app.register_user_main(__main);

  return app.exec();
}

int __main(int argc, char** argv)
{
  const auto filename =
      argc < 2 ? src_path("../../../../data/Boeing_787.obj") : argv[1];

  auto mesh = SimpleTriangleMesh3f{};
  if (!v2::MeshReader().read_object_file(mesh, filename))
  {
    cout << "Cannot reading mesh file:\n" << filename << endl;
    close_window();
    return EXIT_FAILURE;
  }
  cout << "Read " << filename << " successfully" << endl;

  auto w = reinterpret_cast<OpenGLWindow*>(create_gl_window(300, 300));
  display_mesh(mesh);

  // Connect the rotation sliders to the OpenGL window.
  QObject::connect(some_slider, &RotationSliders::sendNewAngles, w,
                   &OpenGLWindow::setEulerAngles);

  for (;;)
  {
    const auto key = get_key();
    if (key == KEY_ESCAPE || key == ' ')
      break;
  }
  close_window();

  return EXIT_SUCCESS;
}
#else  // Normal usage of Qt
int main(int argc, char** argv)
{
  const auto filename =
      argc < 2 ? src_path("../../../../data/Boeing_787.obj") : argv[1];

  auto mesh = SimpleTriangleMesh3f{};
  if (!v2::MeshReader().read_object_file(mesh, filename))
  {
    cout << "Cannot reading mesh file:\n" << filename << endl;
    return EXIT_FAILURE;
  }
  cout << "Read " << filename << " successfully" << endl;

  // Start the GUI.
  QApplication app{argc, argv};
  auto window = OpenGLWindow{512, 512};
  window.setMesh(mesh);

  auto rotation_sliders = RotationSliders{};
  QObject::connect(&rotation_sliders, &RotationSliders::sendNewAngles,  //
                   &window, &OpenGLWindow::setEulerAngles);

  return app.exec();
}
#endif

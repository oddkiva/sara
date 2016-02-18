// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_SARA_GRAPHICS_MESH_HPP
#define DO_SARA_GRAPHICS_MESH_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <DO/Sara/Core/EigenExtension.hpp>


namespace DO { namespace Sara {

  /*!
    \addtogroup Draw3D
    @{
   */

  //! @brief Triangle face consisting of 3 vertex indices
  typedef Array<size_t, 3, 1> Face3;
  //! @brief Quad face consisting of 4 vertex indices
  typedef Array<size_t, 4, 1> Face4;

  //! @brief Simple mesh data structure.
  template <typename Vector_, typename Face_>
  class SimpleMesh
  {
  public:
    typedef Vector_ Vector, Point;
    typedef Face_ Face;

    std::vector<Point>& vertices() { return v_; }
    std::vector<Vector>& normals() { return n_; }
    std::vector<Face>& faces() { return f_; }
    Point& vertex(size_t i) { return v_[i]; }
    Vector& normal(size_t i) { return n_[i]; }
    Face& face(size_t i) { return f_[i]; }
    Point& vertex_of_face(size_t v, size_t f)
    { return vertex(face(f)(v)); }
    Vector& normal_of_vertex_of_face(size_t v, size_t f)
    { return normal(face(f)(v)); }

    const std::vector<Point>& vertices() const { return v_; }
    const std::vector<Vector>& normals() const { return n_; }
    const std::vector<Face>& faces() const { return f_; }
    const Point& vertex(size_t i) const { return v_[i]; }
    const Vector& normal(size_t i) const { return n_[i]; }
    const Face& face(size_t i) const { return f_[i]; }
    const Point& vertex_of_face(size_t v, size_t f) const
    { return vertex(face(f)(v)); }
    const Vector& normal_of_vertex_of_face(size_t v, size_t f) const
    { return normal(face(f)(v)); }

    Point center() const
    {
      Point c(Vector::Zero());
      for (size_t i = 0; i != v_.size(); ++i)
        c += v_[i];
      c /= static_cast<float>(v_.size());
      return c;
    }
    Vector face_normal(size_t f) const
    {
      Vector u(vertex_of_face(1, f) - vertex_of_face(0, f));
      Vector v(vertex_of_face(2, f) - vertex_of_face(0, f));
      Vector n = u.cross(v);
      n.normalize();
      return n;
    }

  private:
    std::vector<Point> v_;
    std::vector<Vector> n_;
    std::vector<Face> f_;
  };

  //! @brief Simple mesh data structure that should be used preferably for
  //! OpenGL.
  typedef SimpleMesh<Point3f, Face3> SimpleTriangleMesh3f;

  //! @brief Mesh reader (WARNING: still experimental!).
  class MeshReader
  {
  public:
    /* WARNING: this function may not work because I just read the vertices and triangles */
    template <typename Vector>
    bool read_object_file(SimpleMesh<Vector, Face3>& mesh,
                          const std::string& fileName)
    {
      // Attempt to read file.
      std::ifstream file(fileName.c_str());
      if(!file)
      {
        std::cerr << "Error reading file!" << std::endl;
        return false;
      }
      // Clear the mesh data structure.
      mesh = SimpleMesh<Vector, Face3>();
      // Fill the mesh data structure.
      std::string line;
      while ( std::getline(file, line) )
      {
        std::stringstream ss;
        ss << line;
        char type;
        ss >> type;
        if(type=='v')
        {
          double x, y, z;
          ss >> x >> y >> z;
          mesh.vertices().push_back(Vector3d(x, y, z).cast<float>());
        }
        if(type=='f')
        {
          size_t a, b, c;
          ss >> a >> b >> c;
          mesh.faces().push_back(Face3(a-1, b-1, c-1));
        }
      }
      // Read mesh successfully.
      computeFaceRings(mesh);
      computeVertexRings(mesh);
      computeNormals(mesh);
      return true;
    }

  private:
    template<typename Vector, typename Face>
    void computeVertexRings(const SimpleMesh<Vector, Face>& mesh)
    {
      vertex_rings_.resize(mesh.vertices().size());
      // Add neighboring vertices.
      for (size_t f = 0; f != mesh.faces().size(); ++f)
      {
        for (int v = 0; v < 3; ++v)
        {
          size_t v1 = mesh.face(f)(  v      );
          size_t v2 = mesh.face(f)( (v+1)%3 );
          size_t v3 = mesh.face(f)( (v+2)%3 );
          vertex_rings_[v1].push_back(v2);
          vertex_rings_[v1].push_back(v3);
        }
      }
      // Eliminate redundancies.
      for (size_t r = 0; r != vertex_rings_.size(); ++r)
      {
        std::vector<size_t>& vertexRing = vertex_rings_[r];
        std::sort(vertexRing.begin(), vertexRing.end());
        typename std::vector<size_t>::iterator
          it = std::unique(vertexRing.begin(), vertexRing.end());
        vertexRing.resize(it-vertexRing.begin());
      }
    }

    template<typename Vector, typename Face>
    void computeFaceRings(const SimpleMesh<Vector, Face>& mesh)
    {
      face_rings_.resize(mesh.vertices().size());
      for (size_t f = 0; f != mesh.faces().size(); ++f)
      {
        face_rings_[mesh.face(f)(0)].push_back(f);
        face_rings_[mesh.face(f)(1)].push_back(f);
        face_rings_[mesh.face(f)(2)].push_back(f);
      }
    }

    template<typename Vector, typename Face>
    Vector computeVertexNormal(const SimpleMesh<Vector, Face>& mesh,
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

    template<typename Vector, typename Face>
    void computeNormals(SimpleMesh<Vector, Face>& mesh)
    {
      mesh.normals().resize(face_rings_.size());
      for(size_t v = 0; v != face_rings_.size(); ++v)
        mesh.normal(v) = computeVertexNormal<Vector>(mesh, v);
    }

  private:
    std::vector<std::vector<size_t> > vertex_rings_;
    std::vector<std::vector<size_t> > face_rings_;
  };

  //! @}

} /* namespace Sara */
} /* namespace DO */


#endif /* DO_SARA_GRAPHICS_MESH_HPP */

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

#include <DO/Sara/Graphics.hpp>

using namespace std;
using namespace DO::Sara;

GRAPHICS_MAIN()
{
  Window w = create_gl_window(300, 300);
  set_active_window(w);

  SimpleTriangleMesh3f mesh;
  string filename = src_path("../../../data/pumpkin_tall_10k.obj");
  if (!MeshReader().read_object_file(mesh, filename))
  {
    cout << "Error reading mesh file:\n" << filename << endl;
    close_window();
    return EXIT_FAILURE;
  }
  cout << "Read " << filename << " successfully" << endl;

  display_mesh(mesh);

  bool quit = false;
  while (!quit)
  {
    int c = get_key();
    quit = (c==KEY_ESCAPE || c==' ');
  }
  close_window();

  return EXIT_SUCCESS;
}

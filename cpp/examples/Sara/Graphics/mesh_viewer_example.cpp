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

int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(__main);
  return app.exec();
}

int __main(int argc, char** argv)
{
  const auto filename =
      argc < 2 ? src_path("../../../../data/pumpkin_tall_10k.obj") : argv[1];

  auto mesh = SimpleTriangleMesh3f{};
  if (!MeshReader().read_object_file(mesh, filename))
  {
    cout << "Cannot reading mesh file:\n" << filename << endl;
    close_window();
    return EXIT_FAILURE;
  }
  cout << "Read " << filename << " successfully" << endl;

  create_gl_window(300, 300);
  display_mesh(mesh);

  for (;;)
  {
    const auto key = get_key();
    if (key == KEY_ESCAPE || key == ' ')
      break;
  }
  close_window();

  return EXIT_SUCCESS;
}

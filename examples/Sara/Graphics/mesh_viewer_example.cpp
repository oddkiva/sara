#include <DO/Graphics.hpp>

using namespace std;
using namespace DO;

GRAPHICS_MAIN()
{
  Window w = create_gl_window(300, 300);
  set_active_window(w);

  SimpleTriangleMesh3f mesh;
  string filename = src_path("../../datasets/pumpkin_tall_10k.obj");
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
#include <DO/Sara/Core/CSV.hpp>
#include <DO/Sara/Graphics.hpp>
#include <DO/Sara/ImageIO.hpp>
#include <DO/Sara/Core/StringFormat.hpp>

#include <algorithm>

#include <array>


namespace sara = DO::Sara;


struct Quad
{
  std::array<Eigen::Vector2f, 4> pixel_coordinates;
  Eigen::Vector3f world_coordinates;
};

// Read data from from CSV.
int parse_quad(int argc, char** argv)
{
  if (argc < 2)
    return 1;

  const auto image_filepath = std::string{argv[1]};
  const auto image = sara::imread<sara::Rgb8>(image_filepath);

  const auto quad_filepath = std::string{argv[2]};
  const auto quads = sara::from_csv<Quad>(  //
      quad_filepath,
      [](const std::vector<std::string>& row) -> Quad {
        auto q = Quad{};
        // Parse the quads
        for (auto i = 0; i < 4; ++i)
          q.pixel_coordinates[i] << std::stof(row[i * 2]),
              std::stof(row[i * 2 + 1]);

        q.world_coordinates << std::stof(row[8]), std::stof(row[9]),
            std::stof(row[10]);

        return q;
      },
      " "  //
  );

  sara::create_window(image.sizes());
  sara::display(image);
  sara::set_antialiasing();
  for (const auto& q : quads)
  {
    const auto& v = q.pixel_coordinates;
    for (auto i = 0; i < 4; ++i)
      sara::draw_line(v[i], v[(i + 1) % 4], sara::Yellow8);

    const Eigen::Vector2f center =
        std::accumulate(q.pixel_coordinates.begin(), q.pixel_coordinates.end(),
                        Eigen::Vector2f{0, 0},
                        [](const auto& a, const auto& b) { return a + b; }) /
        4.f;

    sara::fill_circle(center, 5.f, sara::Black8);
    sara::draw_text(
        center.x(), center.y() + 10,
        sara::format("x=%0.2f y=%0.2f z=%0.2f", q.world_coordinates.x(),
                     q.world_coordinates.y(), q.world_coordinates.z()),
        sara::White8, 20, 0, false, true);
  }

  sara::get_key();

  return 0;
}

int main(int argc, char** argv)
{
  DO::Sara::GraphicsApplication app(argc, argv);
  app.register_user_main(parse_quad);
  return app.exec();
}

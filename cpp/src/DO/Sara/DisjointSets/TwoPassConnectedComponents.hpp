#pragma once

#include <DO/Sara/DisjointSets/DisjointSets.hpp>


namespace DO { namespace Sara {

  auto two_pass_connected_components(const ImageView<int, 2>& values)
      -> Image<int, 2>
  {
    auto labels = Image<int, 2>{values.sizes()};
    labels.flat_array().fill(-1);

    // We don't need the adjacency list data but the API requires one. So we
    // create an empty adjacency list.
    auto adj_list_data = std::vector<std::vector<std::size_t>>{};
    auto adj_list = AdjacencyList(adj_list_data);

    // The disjoint sets is the data structure that we need.
    DisjointSets ds(values.size(), adj_list);

    // Useful variables and functions.
    constexpr auto bg = 0;
    const auto w = values.width();
    const auto h = values.height();
    auto vertex = [&](int x, int y) { return y * w + x; };

    auto last_label_id = 0;

    // First pass.
    for (int y = 0; y < values.height(); ++y)
    {
      for (int x = 0; x < values.width(); ++x)
      {
        // e = (x, y) is a non-background pixel.

        // b = (x, y - 1) is the north pixel.
        if (y > 0 && values(x, y - 1) == values(x, y))
          labels(x, y) = labels(x, y - 1);

        // c = (x + 1, y - 1).
        else if (x + 1 < w & y > 0 && values(x + 1, y - 1) == values(x, y))
        {
          const auto c = labels(x + 1, y - 1);
          labels(x, y) = c;

          // a = (x - 1, y - 1).
          if (x > 0 && values(x - 1, y - 1) == values(x, y))
          {
            const auto d = labels(x - 1, y - 1);
            labels(x, y) = d;
          }

          else if (x > 0 && values(x - 1, y) == values(x, y))
          {
            const auto d = labels(x - 1, y);
            ds.join(ds.node(c), ds.node(d));
          }
        }

        else if (x > 0 && y > 0 && values(x - 1, y - 1) == values(x, y))
          labels(x, y) = labels(x - 1, y - 1);

        else if (x > 0 && values(x - 1, y) == values(x, y))
          labels(x, y) = labels(x - 1, y);

        else
        {
          ds.make_set(last_label_id);
          labels(x, y) = last_label_id;
          ++last_label_id;
        }
      }
    }


    for (int y = 0; y < values.height(); ++y)
      for (int x = 0; x < values.width(); ++x)
        labels(x, y) = ds.component(labels(x, y));

    return labels;
  }

} /* namespace Sara */
} /* namespace DO */

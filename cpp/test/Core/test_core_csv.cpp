#define BOOST_TEST_MODULE "CSV I/O"

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/CSV.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>


BOOST_AUTO_TEST_CASE(test_csv_read_write)
{
  using namespace DO::Sara;
  using tuple_type = std::tuple<int, std::string>;

  {
    auto array = std::vector<tuple_type>{{std::make_tuple(0, "zero"),  //
                                          std::make_tuple(1, "one")}};
    to_csv(array, "/home/david/Desktop/test.csv");
  }

  {
    auto array = from_csv<tuple_type>(
        "/home/david/Desktop/test.csv",
        [](const std::vector<std::string>& row) -> tuple_type {
          return {std::stoi(row[0]), row[1]};
        });

    for (const auto& t: array)
    {
      std::cout << std::get<0>(t) << " " << std::get<1>(t) << std::endl;
    }
  }
}

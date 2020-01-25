#define BOOST_TEST_MODULE "CSV I/O"

#include <DO/Sara/Core/EigenExtension.hpp>
#include <DO/Sara/Core/CSV.hpp>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>


namespace fs = boost::filesystem;


BOOST_AUTO_TEST_CASE(test_csv_read_write)
{
  using namespace DO::Sara;
  using tuple_type = std::tuple<int, std::string>;

  const auto filepath = (fs::temp_directory_path() / "test.csv").string();

  // Write data to CSV.
  {
    const auto array = std::vector<tuple_type>{{std::make_tuple(0, "zero"),  //
                                                std::make_tuple(1, "one")}};
    to_csv(array, filepath);
  }

  // Read data from from CSV.
  {
    const auto array = from_csv<tuple_type>(
        filepath, [](const std::vector<std::string>& row) -> tuple_type {
          return {std::stoi(row[0]), row[1]};
        });

    const auto& [i0, s0] = array[0];
    const auto& [i1, s1] = array[1];

    BOOST_CHECK_EQUAL(i0, 0);
    BOOST_CHECK_EQUAL(s0, "zero");
    BOOST_CHECK_EQUAL(i1, 1);
    BOOST_CHECK_EQUAL(s1, "one");
  }

  fs::remove(filepath);
}

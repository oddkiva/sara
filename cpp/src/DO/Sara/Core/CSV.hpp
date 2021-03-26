#pragma once

#include <boost/algorithm/string.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>


namespace DO::Sara {

  //! @ingroup Core
  //! @defgroup CSV CSV  I/O functions
  //! @{

  inline auto split(const std::string& line, const std::string& delimiters)
      -> std::vector<std::string>
  {
    auto tokens = std::vector<std::string>{};
    boost::split(tokens, line, boost::is_any_of(delimiters));
    return tokens;
  }


  template <typename Tuple>
  auto
  from_csv(const std::string& filename,
           std::function<Tuple(const std::vector<std::string>&)> parse_row_fn,
           const std::string& delimiters = ",") -> std::vector<Tuple>
  {
    std::ifstream file{filename};
    if (!file)
      throw std::runtime_error{"Could not read CSV file!"};

    auto array = std::vector<Tuple>{};

    auto line = std::string{};
    auto line_index = 0;
    while (std::getline(file, line))
    {
      const auto row = split(line, delimiters);
      if (line_index == 0)
      {
        try
        {
          auto row_parsed = parse_row_fn(row);
          array.emplace_back(row_parsed);
        }
        catch (...)
        {
#ifdef DEBUG
          std::cout << "Ignore header: " << std::endl;
          for (const auto& c : row)
            std::cout << c << " ";
          std::cout << std::endl;
#endif
        }
      }
      else
      {
        auto row_parsed = parse_row_fn(row);
        array.emplace_back(row_parsed);
      }
      ++line_index;
    }

    return array;
  }


  template <typename>
  struct is_tuple : std::false_type
  {
  };

  template <typename... T>
  struct is_tuple<std::tuple<T...>> : std::true_type
  {
  };


  template <typename Tuple>
  auto to_csv(const std::vector<Tuple>& array, const std::string& filename)
  {
    std::ofstream file{filename};
    if (!file)
      throw std::runtime_error{"Could not create CSV file!"};

    for (const auto& row : array)
    {
      if constexpr (is_tuple<Tuple>::value)
      {
        std::apply(
            [&file](const auto&... elements) {
              ((file << elements << ","), ...);
              file << "\n";
            },
            row);
      }
      else
        file << row << "\n";
    }
  }

  //! @}

}  // namespace DO::Sara

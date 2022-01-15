#pragma once

#include <drafts/NeuralNetworks/Darknet/Layer.hpp>

#include <boost/filesystem.hpp>


namespace DO::Sara::Darknet {

  struct NetworkParser
  {
    inline auto read_line(std::ifstream& file, std::string& line) const
    {
      if (!std::getline(file, line))
        return false;
      line = boost::algorithm::trim_copy(line);
      return true;
    }

    inline auto is_section(const std::string& line) const
    {
      return line.front() == '[';
    }

    inline auto is_comment(const std::string& line) const
    {
      return line.front() == '#';
    }

    inline auto section_name(const std::string& line) const
    {
      auto line_trimmed = line;
      boost::algorithm::trim_if(
          line_trimmed, [](const auto ch) { return ch == '[' || ch == ']'; });
      return line_trimmed;
    }

    inline auto make_new_layer(const std::string& layer_type,
                               std::vector<std::unique_ptr<Layer>>& nodes) const
    {
      std::cout << "MAKING NEW LAYER: " << layer_type << std::endl;

      if (layer_type == "net")
        nodes.emplace_back(new Input);
      else if (layer_type == "convolutional")
        nodes.emplace_back(new Convolution);
      else if (layer_type == "route")
        nodes.emplace_back(new Route);
      else if (layer_type == "maxpool")
        nodes.emplace_back(new MaxPool);
      else if (layer_type == "upsample")
        nodes.emplace_back(new Upsample);
      else if (layer_type == "yolo")
        nodes.emplace_back(new Yolo);

      nodes.back()->type = layer_type;
    }

    inline auto
    finish_layer_init(std::vector<std::unique_ptr<Layer>>& nodes) const
    {
      const auto& layer_type = nodes.back()->type;
      if (layer_type != "net")
      {
        if (nodes.size() < 2)
          throw std::runtime_error{"Invalid network!"};
        const auto& previous_node = *(nodes.rbegin() + 1);
        nodes.back()->input_sizes = previous_node->output_sizes;
      }

      if (layer_type == "net")
        dynamic_cast<Input&>(*nodes.back()).update_output_sizes();
      else if (layer_type == "convolutional")
        dynamic_cast<Convolution&>(*nodes.back()).update_output_sizes();
      else if (layer_type == "route")
        dynamic_cast<Route&>(*nodes.back()).update_output_sizes(nodes);
      else if (layer_type == "maxpool")
        dynamic_cast<MaxPool&>(*nodes.back()).update_output_sizes();
      else if (layer_type == "upsample")
        dynamic_cast<Upsample&>(*nodes.back()).update_output_sizes();
      else if (layer_type == "yolo")
        dynamic_cast<Yolo&>(*nodes.back()).update_output_sizes(nodes);

      std::cout << "CHECKING CURRENT LAYER: " << std::endl;
      std::cout << *nodes.back() << std::endl;
    }

    inline auto parse_config_file(const std::string& cfg_filepath) const
    {
      namespace fs = boost::filesystem;

      const auto data_dir_path =
          fs::canonical(fs::path{src_path("../../../../data")});

      auto file = std::ifstream{cfg_filepath};

      auto line = std::string{};

      auto section = std::string{};
      auto in_current_section = false;
      auto enter_new_section = false;

      auto nodes = std::vector<std::unique_ptr<Layer>>{};

      while (read_line(file, line))
      {
        if (line.empty())
          continue;

        if (is_comment(line))
          continue;

        // Enter a new section.
        if (is_section(line))
        {
          // Finish initialization of the previous layer if there was one.
          if (!section.empty())
            finish_layer_init(nodes);

          // Create a new layer.
          section = section_name(line);
          make_new_layer(section, nodes);

          enter_new_section = true;
          in_current_section = false;
          continue;
        }

        if (enter_new_section)
        {
          in_current_section = true;
          enter_new_section = false;
        }

        if (in_current_section)
          nodes.back()->parse_line(line);
      }

      finish_layer_init(nodes);

      return nodes;
    }
  };

  //! Guaranteed to work only on little-endian machine and 64 bit architecture.
  struct NetworkWeightLoader
  {
    FILE* fp = nullptr;

    int major;
    int minor;
    int revision;
    uint64_t seen;
    int transpose;

    bool debug = false;

    inline NetworkWeightLoader() = default;

    inline NetworkWeightLoader(const std::string& filepath)
    {
      fp = fopen(filepath.c_str(), "rb");
      if (fp == nullptr)
        throw std::runtime_error{"Failed to open file: " + filepath};

      auto num_bytes_read = size_t{};

      num_bytes_read += fread(&major, sizeof(int), 1, fp);
      num_bytes_read += fread(&minor, sizeof(int), 1, fp);
      num_bytes_read += fread(&revision, sizeof(int), 1, fp);
      if ((major * 10 + minor) >= 2)
      {
        if (debug)
          printf("\n seen 64");
        uint64_t iseen = 0;
        num_bytes_read += fread(&iseen, sizeof(uint64_t), 1, fp);
        seen = iseen;
      }
      else
      {
        if (debug)
          printf("\n seen 32");
        uint32_t iseen = 0;
        num_bytes_read += fread(&iseen, sizeof(uint32_t), 1, fp);
        seen = iseen;
      }
      if (debug)
        printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n",
               (float) (seen / 1000), (float) (seen / 64000));
      transpose = (major > 1000) || (minor > 1000);

      // std::cout << "Num bytes read = " << num_bytes_read << std::endl;
    }

    inline ~NetworkWeightLoader()
    {
      if (fp)
      {
        fclose(fp);
        fp = nullptr;
      }
    }

    inline auto load(std::vector<std::unique_ptr<Layer>>& net)
    {
      for (auto& layer : net)
      {
        if (auto d = dynamic_cast<Convolution*>(layer.get()))
        {
          if (debug)
            std::cout << "LOADING WEIGHTS FOR CONVOLUTIONAL LAYER:\n"
                      << *layer << std::endl;
          d->load_weights(fp);
        }
      }
    }
  };


}  // namespace DO::Sara::Darknet

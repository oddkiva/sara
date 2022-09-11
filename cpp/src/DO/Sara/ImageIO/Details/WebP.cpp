#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>
#include <DO/Sara/ImageIO/Details/WebP.hpp>


extern "C" {
#include <webp/decode.h>
#include <webp/encode.h>
}


namespace DO::Sara {

  static constexpr auto webp_status_messages =
      std::array<const char*, VP8_STATUS_NOT_ENOUGH_DATA + 1>{
          "OK",
          "OUT_OF_MEMORY",
          "INVALID_PARAM",
          "BITSTREAM_ERROR",
          "UNSUPPORTED_FEATURE",
          "SUSPENDED",
          "USER_ABORT",
          "NOT_ENOUGH_DATA"  //
      };

  static auto get_webp_error_message(const std::string& in_file, int status)
      -> std::string
  {
    auto oss = std::ostringstream{};
    oss << "Decoding of " + in_file << " failed.\n";
    oss << "Status: " << status << "\n";
    if (status >= VP8_STATUS_OK && status <= VP8_STATUS_NOT_ENOUGH_DATA)
      oss << "(" << webp_status_messages[status] << ")";
    return oss.str();
  }

  static auto read_webp_file(const std::string& filepath)
      -> std::vector<std::uint8_t>
  {
    auto in_file = FileHandle{filepath.c_str(), "rb"};
    if (in_file == nullptr)
      throw std::runtime_error{"Error: cannot open WebP file " + filepath};

    fseek(in_file, 0, SEEK_END);
    const auto file_size = ftell(in_file);
    fseek(in_file, 0, SEEK_SET);

    // we allocate one extra byte for the \0 terminator
    auto image_buffer = std::vector<std::uint8_t>(file_size + 1);

    const auto ok = fread(image_buffer.data(), file_size, 1, in_file) == 1;
    if (!ok)
      throw std::runtime_error{"Could not read " + std::to_string(file_size) +
                               " bytes of data from file " + filepath};

    image_buffer[file_size] = '\0';  // convenient 0-terminator

    return image_buffer;
  }

  static auto load_webp(const std::string& image_filepath,
                        WebPBitstreamFeatures* bitstream)
      -> std::vector<std::uint8_t>
  {
    const auto encoded_image = read_webp_file(image_filepath);

    auto local_features = WebPBitstreamFeatures{};
    if (bitstream == nullptr)
      bitstream = &local_features;

    const auto status = WebPGetFeatures(encoded_image.data(),  //
                                        encoded_image.size(),  //
                                        bitstream);
    if (status != VP8_STATUS_OK)
      throw std::runtime_error{get_webp_error_message(image_filepath, status)};

    return encoded_image;
  }

  static auto initialize_config(WebPDecoderConfig& config, Image<Rgb8>& image)
      -> void
  {
    const auto output_buffer = &config.output;
    auto w = config.input.width;
    auto h = config.input.height;

    if (config.options.use_scaling)
    {
      w = config.options.scaled_width;
      h = config.options.scaled_height;
    }
    else if (config.options.use_cropping)
    {
      w = config.options.crop_width;
      h = config.options.crop_height;
    }

    image.resize(w, h);

    static const auto bytes_per_pixel = 3;
    static const auto stride = bytes_per_pixel * w;
    output_buffer->u.RGBA.stride = stride;
    output_buffer->u.RGBA.size = stride * h;
    output_buffer->u.RGBA.rgba = reinterpret_cast<std::uint8_t*>(image.data());
    output_buffer->is_external_memory = 1;
  }


  auto read_webp_file_as_interleaved_rgb_image(const std::string& filepath)
      -> Image<Rgb8>
  {
    auto image = Image<Rgb8>{};

    // A) Init a configuration object
    auto config = WebPDecoderConfig{};
    if (!WebPInitDecoderConfig(&config))
      throw std::runtime_error{"Failed to initialize WebP decoder config!"};

    auto bitstream = &config.input;
    const auto encoded_image = load_webp(filepath.c_str(), bitstream);

    // B) optional: retrieve the bitstream's features.
    if (WebPGetFeatures(encoded_image.data(), encoded_image.size(),
                        &config.input) != VP8_STATUS_OK)
      throw std::runtime_error{"Failed to retrieve WebP bitstream features!"};

    // C) Adjust 'config' options, if needed
    initialize_config(config, image);

    // E) Decode the full WebP image.
    const auto status = WebPDecode(encoded_image.data(),  //
                                   encoded_image.size(),  //
                                   &config);
    if (status != VP8_STATUS_OK)
      throw std::runtime_error{get_webp_error_message(filepath, status)};

    // F) Decoded image is now in config.output (and config.output.u.RGBA).
    // It can be saved, displayed or otherwise processed.

    // G) Reclaim memory allocated in config's object. It's safe to call
    // this function even if the memory is external and wasn't allocated
    // by WebPDecode().
    WebPFreeDecBuffer(&config.output);

    return image;
  }

  auto write_webp_file(const ImageView<Rgb8>& image,
                       const std::string& filepath, const int quality) -> void
  {
    const auto quality_factor = static_cast<float>(quality / 100.f);
    const auto rgb = reinterpret_cast<const std::uint8_t*>(image.data());

    std::uint8_t* buffer = nullptr;
    const auto buffer_size =
        WebPEncodeRGB(rgb, image.width(), image.height(), image.width() * 3,
                      quality_factor, &buffer);

    auto file_handle = FileHandle{filepath.c_str(), "wb"};
    fwrite(buffer, sizeof(buffer[0]), buffer_size, file_handle);

    free(reinterpret_cast<void*>(buffer));
  }

}  // namespace DO::Sara

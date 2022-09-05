#include <DO/Sara/ImageIO/Details/Heif.hpp>

extern "C" {
#include <libheif/heif.h>
}


namespace DO::Sara {

  auto read_heif_file_as_interleaved_rgb_image(const std::string& filepath)
      -> heif::Image
  {
    auto ctx = heif::Context{};
    ctx.read_from_file(filepath);

    auto image_handle = ctx.get_primary_image_handle();

    const auto image = image_handle.decode_image(heif_colorspace_RGB,
                                                 heif_chroma_interleaved_RGB);
    return image;
  }

  auto to_image_view(heif::Image& image) -> ImageView<Rgb8>
  {
    const auto colorspace = image.get_colorspace();
    if (!(colorspace == heif_colorspace_RGB &&
          image.get_chroma_format() == heif_chroma_interleaved_RGB))
      throw std::runtime_error{
          "Decoded image must in interleaved 24-bit RGB format!"};

    auto stride = int{};
    auto data = reinterpret_cast<Rgb8*>(
        image.get_plane(heif_channel_interleaved, &stride));
    if (data == nullptr)
      return {};

    const auto w = image.get_width(heif_channel_interleaved);
    const auto h = image.get_height(heif_channel_interleaved);
    auto imview = ImageView<Rgb8>{data, {w, h}};
    return imview;
  }

  auto write_heif_file(const ImageView<Rgb8>& image,
                       const std::string& filepath, const int quality) -> void
  {
    const auto w = image.width();
    const auto h = image.height();

    auto error = heif::Error{};

    heif_image* himage = nullptr;
    error = heif_image_create(w, h,                         //
                              heif_colorspace_RGB,          //
                              heif_chroma_interleaved_RGB,  //
                              &himage);
    if (error)
      throw std::runtime_error{error.get_message()};

    error = heif_image_add_plane(himage, heif_channel_interleaved, w, h, 8);
    if (error)
      throw std::runtime_error{error.get_message()};

    // Get the raw data pointer of the heif_image and copy the content of the
    // image view to it.
    auto stride = int{};
    auto data = reinterpret_cast<Rgb8*>(
        heif_image_get_plane(himage, heif_channel_interleaved, &stride));
    std::copy(image.begin(), image.end(), data);

    heif_context* ctx = heif_context_alloc();

    // Use the HEVC codec, which performs best.
    heif_encoder* encoder = nullptr;
    error = heif_context_get_encoder_for_format(ctx, heif_compression_HEVC,
                                                &encoder);
    if (error)
      throw std::runtime_error{error.get_message()};

    // set the encoder parameters
    error = heif_encoder_set_lossy_quality(encoder, quality);
    if (error)
      throw std::runtime_error{error.get_message()};

    if (quality == 100)
      error = heif_encoder_set_lossless(encoder, true);
    if (error)
      throw std::runtime_error{error.get_message()};

    error = heif_context_encode_image(ctx, himage, encoder, nullptr, nullptr);
    if (error)
      throw std::runtime_error{error.get_message()};

    // Free the allocated resources.
    heif_encoder_release(encoder);
    heif_context_write_to_file(ctx, filepath.c_str());
    heif_context_free(ctx);
    heif_image_release(himage);
  }

}  // namespace DO::Sara

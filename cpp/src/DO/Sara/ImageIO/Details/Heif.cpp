#include <DO/Sara/ImageIO/Details/Heif.hpp>

#define ENSURE_HEIF_BACKWARD_COMPATIBILITY
#define USE_C_API

#ifdef USE_C_API
extern "C" {
#  include <libheif/heif.h>
}
#else
#  include <libheif/heif_cxx.h>
#endif


namespace DO::Sara {

  auto read_heif_file_as_interleaved_rgb_image(const std::string& filepath)
      -> Image<Rgb8>
  {
#ifdef USE_C_API
    heif_error error;

    heif_context* ctx = heif_context_alloc();
    error = heif_context_read_from_file(ctx, filepath.c_str(), nullptr);
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    heif_image_handle* handle = nullptr;
    error = heif_context_get_primary_image_handle(ctx, &handle);
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    heif_image* image = nullptr;
    error = heif_decode_image(handle, &image, heif_colorspace_RGB,
                              heif_chroma_interleaved_RGB, nullptr);
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    const auto w = heif_image_get_width(image, heif_channel_interleaved);
    const auto h = heif_image_get_height(image, heif_channel_interleaved);

    auto stride = int{};
    const auto data = reinterpret_cast<Rgb8*>(const_cast<std::uint8_t*>(
        heif_image_get_plane(image, heif_channel_interleaved, &stride)));
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};
    const auto im_view = ImageView<Rgb8>{data, {w, h}};

    auto im = Image<Rgb8>{im_view};

    heif_image_handle_release(handle);
    heif_image_release(image);
    heif_context_free(ctx);

    return im;
#else
    auto ctx = heif::Context{};
    ctx.read_from_file(filepath);

    auto image_handle = ctx.get_primary_image_handle();

    auto image = image_handle.decode_image(heif_colorspace_RGB,
                                           heif_chroma_interleaved_RGB);

    const auto colorspace = image.get_colorspace();
    if (!(colorspace == heif_colorspace_RGB &&
          image.get_chroma_format() == heif_chroma_interleaved_RGB))
      throw std::runtime_error{
          "Decoded image must be in interleaved 24-bit RGB format!"};

    auto stride = int{};
    auto data = reinterpret_cast<Rgb8*>(
        image.get_plane(heif_channel_interleaved, &stride));
    if (data == nullptr)
      return {};

    const auto w = image.get_width(heif_channel_interleaved);
    const auto h = image.get_height(heif_channel_interleaved);
    const auto imview = ImageView<Rgb8>{data, {w, h}};

    auto imcopy = Image<Rgb8>(imview);
    return imcopy;
#endif
  }

  auto write_heif_file(const ImageView<Rgb8>& image,
                       const std::string& filepath, const int quality) -> void
  {
    const auto w = image.width();
    const auto h = image.height();

#ifdef USE_C_API
    auto error = heif_error{};

    heif_image* himage = nullptr;
    error = heif_image_create(w, h,                         //
                              heif_colorspace_RGB,          //
                              heif_chroma_interleaved_RGB,  //
                              &himage);
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

#  ifdef ENSURE_HEIF_BACKWARD_COMPATIBILITY
    error = heif_image_add_plane(himage, heif_channel_interleaved, w, h, 24);
#  else
    error = heif_image_add_plane(himage, heif_channel_interleaved, w, h, 8);
#  endif
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    // Get the raw data pointer of the heif_image and copy the content of the
    // image view to it.
    auto row_byte_size = int{};
    auto dst_ptr = reinterpret_cast<void*>(
        heif_image_get_plane(himage, heif_channel_interleaved, &row_byte_size));
    const auto src_ptr = reinterpret_cast<const void*>(image.data());
    if (row_byte_size != static_cast<int>(sizeof(Rgb8) * image.width()))
      throw std::runtime_error{
          "The row byte size in the HEIF image buffer is incorrect!"};
    std::memcpy(dst_ptr, src_ptr, sizeof(Rgb8) * image.size());

    heif_context* ctx = heif_context_alloc();

    // Use the HEVC codec, which performs best.
    heif_encoder* encoder = nullptr;
#  ifdef ENSURE_HEIF_BACKWARD_COMPATIBILITY
    error = heif_context_get_encoder_for_format(ctx, heif_compression_HEVC,
                                                &encoder);
#  else
    error = heif_context_get_encoder_for_format(nullptr, heif_compression_HEVC,
                                                &encoder);
#  endif
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    // set the encoder parameters
    error = heif_encoder_set_lossy_quality(encoder, quality);
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    if (quality == 100)
      error = heif_encoder_set_lossless(encoder, true);
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    error = heif_context_encode_image(ctx, himage, encoder, nullptr, nullptr);
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};

    // Free the allocated resources.
    heif_encoder_release(encoder);
    error = heif_context_write_to_file(ctx, filepath.c_str());
    if (error.code != heif_error_Ok)
      throw std::runtime_error{error.message};
    heif_context_free(ctx);
    heif_image_release(himage);
#else
    auto himage = heif::Image{};
    himage.create(w, h, heif_colorspace_RGB, heif_chroma_interleaved_RGB);
    himage.add_plane(heif_channel_interleaved, w, h, 8);

    // Get the raw data pointer of the heif_image and copy the content of the
    // image view to it.
    auto stride = int{};
    auto data = reinterpret_cast<Rgb8*>(
        himage.get_plane(heif_channel_interleaved, &stride));
    std::copy(image.begin(), image.end(), data);

    auto encoder = heif::Encoder{heif_compression_HEVC};
    encoder.set_lossy_quality(quality);

    if (quality == 100)
      encoder.set_lossless(true);

    auto ctx = heif::Context{};
    ctx.encode_image(himage, encoder);
    ctx.write_to_file(filepath);
#endif
  }

}  // namespace DO::Sara

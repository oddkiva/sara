// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#if defined(_WIN32) || defined(_WIN32_WCE)
# define NOMINMAX
// We need to include this the include problem with libjpeg in Windows platform.
# include <windows.h>
#endif

#include <DO/Sara/Core.hpp>

#include <DO/Sara/ImageIO/Details/ImageIOObjects.hpp>


using namespace std;


// JPEG I/O.
namespace DO { namespace Sara {

  METHODDEF(void) jpeg_error(j_common_ptr cinfo)
  {
    jpeg_error_message_struct *myerr = (jpeg_error_message_struct *) cinfo->err;
    (*cinfo->err->output_message) (cinfo);
    longjmp(myerr->setjmp_buffer, 1);
  }

  JpegFileReader::JpegFileReader(const char *filepath)
      : _file_handle{filepath, "rb"}
  {
    // Setup error handling.
    _cinfo.err = jpeg_std_error(&_jerr.pub);
    _jerr.pub.error_exit = &jpeg_error;

    if (setjmp(_jerr.setjmp_buffer))
      throw std::runtime_error{
        format("Failed to read file %s", filepath).c_str()};

    // Create decompress structures.
    jpeg_create_decompress(&_cinfo);

    // We are reading a file.
    jpeg_stdio_src(&_cinfo, _file_handle);

    // Read header file.
    if (!jpeg_read_header(&_cinfo, TRUE))
      throw std::runtime_error{
        format("Failed to read JPEG header of file %s", filepath).c_str()};

    // Start reading data.
    if (!jpeg_start_decompress(&_cinfo))
      throw std::runtime_error{
      format("Failed to start JPEG decompression of file %s", filepath).c_str()};
  }

  JpegFileReader::~JpegFileReader()
  {
    jpeg_destroy_decompress(&_cinfo);
  }

  auto JpegFileReader::image_sizes() const -> std::tuple<int, int, int>
  {
    return make_tuple(
      static_cast<int>(_cinfo.output_width),
      static_cast<int>(_cinfo.output_height),
      static_cast<int>(_cinfo.output_components));
  }

  void JpegFileReader::read(unsigned char *data)
  {
    // Scan lines.
    const auto row_stride = _cinfo.output_width * _cinfo.output_components;
    auto row = data;
    while (_cinfo.output_scanline < _cinfo.output_height) {
      JSAMPROW scanline[] = { row };
      jpeg_read_scanlines(&_cinfo, scanline, 1);
      row += row_stride;
    }

    // Wrap up file decompression.
    if (!jpeg_finish_decompress(&_cinfo))
      throw std::runtime_error{"Failed to finish JPEG file decompression"};
  }

  JpegFileWriter::JpegFileWriter(const unsigned char* data, int width,
                                 int height, int depth)
    : _data{data}
  {
    // Setup JPEG error handling.
    _cinfo.err = jpeg_std_error(&_jerr);

    // Create compressor structure.
    jpeg_create_compress(&_cinfo);

    // Image dimensions.
    _cinfo.image_width = width;
    _cinfo.image_height = height;
    _cinfo.input_components = depth;

    // Color space.
    if (_cinfo.input_components == 3)
      _cinfo.in_color_space = JCS_RGB;
    else if (_cinfo.input_components==1)
      _cinfo.in_color_space = JCS_GRAYSCALE;
    else
      throw std::runtime_error{"Unsupported color space for JPEG write"};

    // Prepare writing.
    jpeg_set_defaults(&_cinfo);
  }

  JpegFileWriter::~JpegFileWriter()
  {
    jpeg_destroy_compress(&_cinfo);
  }

  void JpegFileWriter::write(const char *filepath, int quality)
  {
    if (quality < 0 || quality > 100)
      throw std::runtime_error{
          "Error: The JPEG quality parameter must be between 0 and 100"};

    _file_handle.open(filepath, "wb");

    jpeg_stdio_dest(&_cinfo, _file_handle);
    jpeg_set_quality(&_cinfo, quality, TRUE);
    jpeg_start_compress(&_cinfo, TRUE);

    const auto num_bytes_per_line =
        _cinfo.image_width * _cinfo.input_components;

    auto scanline_data = _data;
    auto buffer = vector<JSAMPLE>(num_bytes_per_line);

    while (_cinfo.next_scanline < _cinfo.image_height) {
      JSAMPROW scanline[] = { &buffer[0] };
      copy(scanline_data, scanline_data+num_bytes_per_line, &buffer[0]);
      jpeg_write_scanlines(&_cinfo, scanline, 1);
      scanline_data += num_bytes_per_line;
    }

    jpeg_finish_compress(&_cinfo);
  }

} /* namespace Sara */
} /* namespace DO */


// PNG I/O.
namespace DO { namespace Sara {

  PngFileReader::PngFileReader(const char *filepath)
    : _file_handle{filepath, "rb"}
  {
    _png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!_png_ptr)
      throw std::runtime_error{"Failed to create PNG read structure!"};

    _info_ptr = png_create_info_struct(_png_ptr);
    if (!_info_ptr)
      throw std::runtime_error{"Failed to create PNG info structure!"};

    png_byte header[8];
    if (fread(header, 1, 8, _file_handle) != 8)
      throw std::runtime_error{
        format("Failed to read PNG header in file %s!", filepath).c_str()};

    if (png_sig_cmp(header, 0, 8))
      throw std::runtime_error{
      format("Incorrect PNG signature in file %s!", filepath).c_str()};

    png_init_io(_png_ptr, _file_handle);
    png_set_sig_bytes(_png_ptr, 8);

    png_read_info(_png_ptr, _info_ptr);

    // Get width, height, bit-depth and color type.
    png_get_IHDR(_png_ptr, _info_ptr, &_width, &_height, &_bit_depth,
                 &_color_type, NULL, NULL, NULL);

    // Expand images of all color-type to 8-bit.
    if (_color_type == PNG_COLOR_TYPE_PALETTE)
      png_set_expand(_png_ptr);
    if (_bit_depth < 8)
      png_set_expand(_png_ptr);
    if (png_get_valid(_png_ptr, _info_ptr, PNG_INFO_tRNS))
      png_set_expand(_png_ptr);
    if (_bit_depth == 16) // convert 16-bit to 8-bit on the fly
      png_set_strip_16(_png_ptr);

    // If required, set the gamma conversion.
    double gamma;
    if (png_get_gAMA(_png_ptr, _info_ptr, &gamma))
      png_set_gamma(_png_ptr, 2.2, gamma);

    // The transformations are now registered, so update _info_ptr data.
    png_read_update_info(_png_ptr, _info_ptr);

    // Update width, height and new bit-depth and color type.
    png_get_IHDR(_png_ptr, _info_ptr, &_width, &_height, &_bit_depth,
                 &_color_type, NULL, NULL, NULL);

    _channels = png_get_channels(_png_ptr, _info_ptr);
  }

  PngFileReader::~PngFileReader()
  {
    png_destroy_read_struct(&_png_ptr, &_info_ptr, NULL);
  }

  auto PngFileReader::image_sizes() const -> std::tuple<int, int, int>
  {
    return make_tuple(
      static_cast<int>(_width),
      static_cast<int>(_height),
      static_cast<int>(_channels)
    );
  }

  void PngFileReader::read(unsigned char *data)
  {
    // Now we can safely get the data correctly.
    png_uint_32 rowbytes = (png_uint_32) png_get_rowbytes(_png_ptr, _info_ptr);

    vector<png_bytep> row_pointers(_width * _height);
    for (auto y = 0u; y < _height; ++y)
      row_pointers[y] = static_cast<png_byte *>(data) + rowbytes*y;

    png_read_image(_png_ptr, &row_pointers[0]);
    png_read_end(_png_ptr, NULL);
  }

  PngFileWriter::PngFileWriter(const unsigned char* data, int width, int height,
                               int depth)
    : _data{data}
    , _width{width}
    , _height{height}
    , _depth{depth}
  {
    _png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                      NULL, NULL, NULL);
    if (!_png_ptr)
      throw 0;

    _info_ptr = png_create_info_struct(_png_ptr);
    if (!_info_ptr)
      throw 0;
  }

  PngFileWriter::~PngFileWriter()
  {
    png_destroy_write_struct(&_png_ptr, &_info_ptr);
  }

  void PngFileWriter::write(const char *filepath)
  {
    _file_handle.open(filepath, "wb");

    png_init_io(_png_ptr, _file_handle);

    // Color spaces are defined at png.h:841+.
    auto color_space = char{};
    switch(_depth)
    {
    case 4: color_space = PNG_COLOR_TYPE_RGBA;
      break;
    case 3: color_space = PNG_COLOR_TYPE_RGB;
      break;
    case 1: color_space = PNG_COLOR_TYPE_GRAY;
      break;
    default:
      throw std::runtime_error{"Invalid color space!"};
    }

    constexpr auto bit_depth = 8;
    png_set_IHDR(_png_ptr, _info_ptr, _width, _height, bit_depth, color_space,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE,
                 PNG_FILTER_TYPE_BASE);
    png_write_info(_png_ptr, _info_ptr);

    auto row_pointers = vector<png_bytep>(_depth * _height);
    for (int y = 0; y < _height; ++y)
      row_pointers[y] = (png_bytep) _data + _width * _depth * y;

    png_write_image(_png_ptr, &row_pointers[0]);
    png_write_end(_png_ptr, NULL);
  }

} /* namespace Sara */
} /* namespace DO */


// Tiff I/O.
namespace DO { namespace Sara {

  TiffFileReader::TiffFileReader(const char *filepath)
    : _file_handle{filepath, "r"}
  {
    _tiff = TIFFOpen(filepath, "r");
    if (!_tiff)
      throw std::runtime_error{
        format("Failed to open file %s", filepath).c_str()};

    TIFFGetField(_tiff, TIFFTAG_IMAGEWIDTH, &_width);
    TIFFGetField(_tiff, TIFFTAG_IMAGELENGTH, &_height);
  }

  TiffFileReader::~TiffFileReader()
  {
    if (_tiff)
      TIFFClose(_tiff);
  }

  auto TiffFileReader::image_sizes() const -> std::tuple<int, int, int>
  {
    constexpr auto depth = 4;
    return make_tuple(_width, _height, depth);
  }

  void TiffFileReader::read(unsigned char *data)
  {
    TIFFReadRGBAImageOriented(_tiff, _width, _height,
                              reinterpret_cast<uint32*>(data),
                              ORIENTATION_TOPLEFT, 0);
  }

  TiffFileWriter::TiffFileWriter(const unsigned char* data, int width,
                                 int height, int depth)
    : _data{data}
    , _width{width}
    , _height{height}
    , _depth{depth}
  {
  }

  void TiffFileWriter::write(const char *filepath)
  {
    auto out = TIFFOpen(filepath, "w");
    if (out == NULL)
      throw std::runtime_error{"Could not create image file"};

    // We need to set some values for basic tags before we can add any data
    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, _width);
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, _height);

    constexpr auto bit_depth = 8;
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, bit_depth);
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, _depth);
    TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    // Write the information to the file
    TIFFWriteEncodedStrip(out, 0, const_cast<unsigned char*>(&_data[0]),
                          _width * _height * _depth);

    TIFFClose(out);
  }

} /* namespace Sara */
} /* namespace DO */

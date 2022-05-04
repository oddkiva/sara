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

#pragma once

#include <DO/Sara/Core/StringFormat.hpp>

extern "C" {
#include <jpeglib.h>
#include <png.h>
#include <setjmp.h>
#ifndef __EMSCRIPTEN__
#  include <tiffio.h>
#endif
}

#include <array>
#include <exception>
#include <stdexcept>


// Base classes for image reading and writing.
namespace DO { namespace Sara {

  //! @addtogroup ImageIO
  //! @{

  //! @brief FileHandle class.
  class FileHandle
  {
    FILE* _file{nullptr};

  public:
    FileHandle() = default;

    FileHandle(const char* filepath, const char* mode)
    {
      open(filepath, mode);
    }

    ~FileHandle()
    {
      close();
    }

    void open(const char* filepath, const char* mode)
    {
#ifdef WIN32
      if (fopen_s(&_file, filepath, mode) != 0)
#else
      _file = fopen(filepath, mode);
      if (_file == nullptr)
#endif
        throw std::runtime_error{
            format("Error opening file '%s' in mode %s!", filepath, mode)
                .c_str()};
    }

    void close()
    {
      if (_file)
        fclose(_file);
    }

    operator FILE*()
    {
      return _file;
    }
  };

  //! @}

}}  // namespace DO::Sara


// Jpeg I/O.
namespace DO { namespace Sara {

  //! @addtogroup ImageIO
  //! @{

  //! @brief JPEG error message.
  struct jpeg_error_message_struct
  {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
  };

  //! @brief JPEG image reader class.
  class JpegFileReader
  {
    //! @{
    //! @brief Decompressor structures.
    jpeg_decompress_struct _cinfo;
    jpeg_error_message_struct _jerr;
    //! @}

    //! @brief File handle.
    FileHandle _file_handle;

  public:
    JpegFileReader() = delete;

    JpegFileReader(const JpegFileReader&) = delete;

    JpegFileReader(const char* filepath);

    JpegFileReader& operator=(const JpegFileReader&) = delete;

    ~JpegFileReader();

    auto image_sizes() const -> std::tuple<int, int, int>;

    void read(unsigned char* data);
  };

  //! @brief JPEG image writer class.
  class JpegFileWriter
  {
    //! @{
    //! @brief Compressor structure.
    jpeg_compress_struct _cinfo;
    jpeg_error_mgr _jerr;
    //! @}

    FileHandle _file_handle;

    //! @brief Image data.
    const unsigned char* _data;

  public:
    JpegFileWriter() = delete;

    JpegFileWriter(const JpegFileWriter&) = delete;

    JpegFileWriter(const unsigned char* data, int width, int height, int depth);

    JpegFileWriter& operator=(const JpegFileWriter&) = delete;

    ~JpegFileWriter();

    void write(const char* filepath, int quality);
  };

  //! @}

}}  // namespace DO::Sara


// PNG I/O.
namespace DO { namespace Sara {

  //! @addtogroup ImageIO
  //! @{

  //! @brief PNG image reader class.
  class PngFileReader
  {
    png_structp _png_ptr;
    png_infop _info_ptr;

    FileHandle _file_handle;

    png_uint_32 _width, _height;
    png_byte _channels;
    int _bit_depth;
    int _color_type;

  public:
    PngFileReader() = delete;

    PngFileReader(const PngFileReader&) = delete;

    PngFileReader(const char* filepath);

    PngFileReader& operator=(const PngFileReader&) = delete;

    ~PngFileReader();

    auto image_sizes() const -> std::tuple<int, int, int>;

    void read(unsigned char* data);
  };

  //! @brief PNG image writer class.
  class PngFileWriter
  {
    png_structp _png_ptr;
    png_infop _info_ptr;

    //! @brief File handle.
    FileHandle _file_handle;

    //! @brief Image data.
    const unsigned char* _data;
    int _width;
    int _height;
    int _depth;

  public:
    PngFileWriter() = delete;

    PngFileWriter(const PngFileWriter&) = delete;

    PngFileWriter(const unsigned char* data, int width, int height, int depth);

    PngFileReader& operator=(const PngFileReader&) = delete;

    ~PngFileWriter();

    void write(const char* filepath);
  };

  //! @}

}}  // namespace DO::Sara


#ifndef __EMSCRIPTEN__
// Tiff I/O.
namespace DO::Sara {

  //! @addtogroup ImageIO
  //! @{

  //! @brief TIFF image reader class.
  class TiffFileReader
  {
    TIFF* _tiff;
    std::uint32_t _width, _height;

  public:
    TiffFileReader(const char* filepath);

    ~TiffFileReader();

    auto image_sizes() const -> std::tuple<int, int, int>;

    void read(unsigned char* data);
  };

  //! @brief TIFF image writer class.
  class TiffFileWriter
  {
    //! @brief Image data.
    const unsigned char* _data;

    //! @brief Image sizes.
    int _width;
    int _height;
    int _depth;

  public:
    TiffFileWriter(const unsigned char* data, int width, int height, int depth);

    void write(const char* filepath);
  };

  //! @}

}  // namespace DO::Sara
#endif

//! @}

#include "ImageIO.hpp"
#include <vector>

using namespace std;

FileError::FileError(const string& filepath, const string& mode)
  : filepath_(filepath), mode_(mode)
{
}

const char * FileError::what() const throw()
{
  ostringstream os;
  os
    << "Error: cannot read file:" << endl << filepath_ << endl
    << "in mode " << mode_ << endl;
  return os.str().c_str();
}

FileHandler::FileHandler(const string& filepath,
                         const string& mode)
{
  file_ = fopen(filepath.c_str(), mode.c_str());
  if (!file_)
    throw FileError(filepath, mode);
}

FileHandler::~FileHandler()
{
  if (file_)
    fclose(file_);
}

METHODDEF(void) jpeg_error(j_common_ptr cinfo)
{
  JpegErrorMessage *myerr = (JpegErrorMessage *) cinfo->err;
  (*cinfo->err->output_message) (cinfo);
  longjmp(myerr->setjmp_buffer, 1);
}

JpegFileReader::JpegFileReader(const string& filepath)
  : ImageFileReader(filepath, "rb")
{
  cinfo_.err = jpeg_std_error(&jerr_.pub);
  jerr_.pub.error_exit = &jpeg_error;

  if (setjmp(jerr_.setjmp_buffer))
    throw FileError(filepath, "rb");

  jpeg_create_decompress(&cinfo_);
  jpeg_stdio_src(&cinfo_, file_);
}

JpegFileReader::~JpegFileReader()
{
  jpeg_destroy_decompress(&cinfo_);
}

bool JpegFileReader::operator()(unsigned char *& data,
                                int& width, int& height, int& depth)
{
  // Read header file.
  if (!jpeg_read_header(&cinfo_, TRUE))
    return false;
  // Start reading data.
  if (!jpeg_start_decompress(&cinfo_))
    return false;
  // Allocate image data.
  width = cinfo_.output_width;
  height = cinfo_.output_height;
  depth = cinfo_.output_components;
  data = new unsigned char[width*height*depth];
  // Scan lines.
  const int row_stride = width * depth;
  unsigned char *row = data;
  while (cinfo_.output_scanline < cinfo_.output_height) {
    JSAMPROW scanline[] = { row };
    jpeg_read_scanlines(&cinfo_, scanline, 1);
    row += row_stride;
  }
  // Stop reading data.
  if (!jpeg_finish_decompress(&cinfo_))
    return false;

  return true;
}

JpegFileWriter::JpegFileWriter(const unsigned char *data,
                               int width, int height, int depth)
  : ImageFileWriter(data, width, height, depth)
{
  cinfo_.err = jpeg_std_error(&jerr_);
  jpeg_create_compress(&cinfo_);
  // Image dimensions.
  cinfo_.image_width = width_;
  cinfo_.image_height = height_;
  cinfo_.input_components = depth_;
  // Color space.
  if (cinfo_.input_components == 3)
    cinfo_.in_color_space = JCS_RGB;
  else if (cinfo_.input_components==1)
    cinfo_.in_color_space = JCS_GRAYSCALE;
  else
  {
    cerr << "Error: Unsupported number of components" << endl;
    throw 0;
  }
  // Prepare writing.
  jpeg_set_defaults(&cinfo_);
}

JpegFileWriter::~JpegFileWriter()
{
  jpeg_destroy_compress(&cinfo_);
}

bool JpegFileWriter::operator()(const string& filepath,
                                int quality)
{
  if (quality < 0 || quality > 100)
  {
    cerr << "Error: The quality parameter should be between 0 and 100" << endl;
    return false;
  }

  file_ = fopen(filepath.c_str(), "wb");
  if (!file_)
    return false;

  jpeg_stdio_dest(&cinfo_, file_);
  jpeg_set_quality(&cinfo_, quality, TRUE);
  jpeg_start_compress(&cinfo_, TRUE);

  const unsigned char *scanline_data = data_;
  const int num_bytes_per_line = width_*depth_;
  vector<JSAMPLE> buffer(num_bytes_per_line);
  while (cinfo_.next_scanline < cinfo_.image_height) {
    JSAMPROW scanline[] = { &buffer[0] };
    copy(scanline_data, scanline_data+num_bytes_per_line, &buffer[0]);
    jpeg_write_scanlines(&cinfo_, scanline, 1);
    scanline_data += num_bytes_per_line;
  }

  jpeg_finish_compress(&cinfo_);

  return true;
}

PngFileReader::PngFileReader(const string& filepath)
  : ImageFileReader(filepath, "rb")
{
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 
                                               NULL, NULL, NULL);
  if (!png_ptr)
    throw FileError(filepath, "rb");

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    throw FileError(filepath, "rb");
}

PngFileReader::~PngFileReader()
{
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
}

bool PngFileReader::operator()(unsigned char *& data,
                               int& width, int& height, int& depth)
{
  png_byte header[8];
  if (fread(header, 1, 8, file_) != 8)
    cerr << "fread failed." << endl;
  if (png_sig_cmp(header, 0, 8))
    return false;

  png_init_io(png_ptr, file_);
  png_set_sig_bytes(png_ptr, 8);
  
  png_read_info(png_ptr, info_ptr);

  png_uint_32 pngWidth, pngHeight;
  int bitDepth, colorType, interlaceType;
  png_get_IHDR(png_ptr, info_ptr, &pngWidth, &pngHeight, &bitDepth, &colorType,
               &interlaceType, (int *)NULL, (int *)NULL);

  png_read_update_info(png_ptr, info_ptr);
  png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
  png_byte channels = png_get_channels(png_ptr, info_ptr);

  width = static_cast<int>(pngWidth);
  height = static_cast<int>(pngHeight);
  depth = static_cast<int>(channels);
  data = new unsigned char[width*height*depth];

  vector<png_bytep> row_pointers(width*height);
  for (png_uint_32 y = 0; y < height; ++y)
    row_pointers[y] = static_cast<png_byte *>(data) + rowbytes*y;

  png_read_image(png_ptr, &row_pointers[0]);
  png_read_end(png_ptr, NULL);
  
  return true;
}

PngFileWriter::PngFileWriter(const unsigned char *data,
                             int width, int height, int depth)
  : ImageFileWriter(data, width, height, depth)
{
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                    NULL, NULL, NULL);
  if (!png_ptr)
    throw 0;

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    throw 0;
}

PngFileWriter::~PngFileWriter()
{
  png_destroy_write_struct(&png_ptr, &info_ptr);
}

bool PngFileWriter::operator()(const string& filepath,
                               int quality)
{
  file_ = fopen(filepath.c_str(), "wb");
  if (!file_)
    return false;

  png_init_io(png_ptr, file_);

  // Color spaces are defined at png.h:841+.
  char color_space;
  switch(depth_)
  {
  case 4: color_space = PNG_COLOR_TYPE_RGBA;
    break;
  case 3: color_space = PNG_COLOR_TYPE_RGB;
    break;
  case 1: color_space = PNG_COLOR_TYPE_GRAY;
    break;
  default:
    return false;
  }

  png_set_IHDR(png_ptr, info_ptr, width_, height_,
               8, color_space, PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  png_write_info(png_ptr, info_ptr);

  vector<png_bytep> row_pointers(depth_*height_);
  for (int y = 0; y < height_; ++y)
    row_pointers[y] = (png_bytep) data_ + width_*depth_*y;

  png_write_image(png_ptr, &row_pointers[0]);
  png_write_end(png_ptr, NULL);
  
  return true;
}

TiffFileReader::TiffFileReader(const std::string& filepath)
  : ImageFileReader(filepath, "r")
{
  tiff_ = TIFFOpen(filepath.c_str(), "r");
  if (!tiff_)
    throw FileError(filepath, "r");
}

TiffFileReader::~TiffFileReader()
{
  if (tiff_)
    TIFFClose(tiff_);
}

bool TiffFileReader::operator()(unsigned char *& data,
                                int& width, int& height, int& depth)
{
  uint32 w, h;
  TIFFGetField(tiff_, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(tiff_, TIFFTAG_IMAGELENGTH, &h);
  width = w;
  height = h;
  depth = 4;
  data = new unsigned char[width*height*depth];
  TIFFReadRGBAImage(tiff_, w, h, reinterpret_cast<uint32 *>(data), 0);
  return true;
}

TiffFileWriter::TiffFileWriter(const unsigned char *data,
                               int width, int height, int depth)
  : ImageFileWriter(data, width, height, depth), out(NULL)
{
}

TiffFileWriter::~TiffFileWriter()
{
  if (out)
    TIFFClose(out);
}

bool TiffFileWriter::operator()(const std::string& filepath, int quality)
{
  // Open the TIFF file
  if((out = TIFFOpen(filepath.c_str(), "w")) == NULL){
    std::cerr << "Unable to write tif file: " << filepath << std::endl;
    return false;
  }
  
  // We need to set some values for basic tags before we can add any data
  TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width_);
  TIFFSetField(out, TIFFTAG_IMAGELENGTH, height_);
  TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, depth_);
  TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  
  TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
  TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  
  // Write the information to the file
  TIFFWriteEncodedStrip(out, 0, const_cast<unsigned char *>(&data_[0]),
                        width_*height_*depth_);
  
  return true;
}
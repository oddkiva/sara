#include <cstdio>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

extern "C" {
# include <libjpeg/jpeglib.h>
# include <libpng/png.h>
# include <setjmp.h>
}

class FileError: public std::exception
{
  std::string filepath_;
  std::string mode_;
public:
  FileError(const std::string& filepath, const std::string& mode);
  virtual const char* what() const throw();
};

class FileReader
{
protected:
  FILE* file_;
public:
  FileReader(const std::string& filepath, const std::string& mode);
  virtual ~FileReader();
};

struct JpegErrorMessage {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

class JpegFileReader : public FileReader
{
  struct jpeg_decompress_struct cinfo_;
  struct JpegErrorMessage jerr_;
  using FileReader::file_;
public:
  //! TODO: make better exception...
  JpegFileReader(const std::string& filepath);
  ~JpegFileReader();
  bool read(unsigned char *& data, int& width, int& height, int& depth);
};


int WriteJpgStream(FILE *file,
  const vector<unsigned char> & array,
  int w,
  int h,
  int depth,
  int quality) {
    if (quality < 0 || quality > 100)
      cerr << "Error: The quality parameter should be between 0 and 100";

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = depth;

    if (cinfo.input_components==3) {
      cinfo.in_color_space = JCS_RGB;
    } else if (cinfo.input_components==1) {
      cinfo.in_color_space = JCS_GRAYSCALE;
    } else {
      cerr << "Error: Unsupported number of channels in file";
      jpeg_destroy_compress(&cinfo);
      return 0;
    }

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    const unsigned char *ptr = &array[0];
    int row_bytes = cinfo.image_width*cinfo.input_components;

    JSAMPLE *row = new JSAMPLE[row_bytes];

    while (cinfo.next_scanline < cinfo.image_height) {
      memcpy(&row[0], &ptr[0], row_bytes*sizeof(unsigned char));
      //int i;
      //for (i = 0; i < row_bytes; ++i)
      //	row[i] = ptr[i];
      jpeg_write_scanlines(&cinfo, &row, 1);
      ptr += row_bytes;
    }

    delete [] row;

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    return 1;
}

int ReadPngStream(FILE *file,
  vector<unsigned char> * ptr,
  int * w,
  int * h,
  int * depth)  {
    png_byte header[8];

    if (fread(header, 1, 8, file) != 8) {
      cerr << "fread failed.";
    }
    if (png_sig_cmp(header, 0, 8))
      return 0;

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
      NULL, NULL, NULL);

    if (!png_ptr)
      return 0;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)  {
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
      return 0;
    }

    png_init_io(png_ptr, file);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    png_uint_32 pngWidth, pngHeight;
    int bitDepth, colorType, interlaceType;
    png_get_IHDR(png_ptr, info_ptr, &pngWidth, &pngHeight, &bitDepth, &colorType,
      &interlaceType, (int*)NULL, (int*)NULL);

    png_read_update_info(png_ptr, info_ptr);
    png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    int channels = (int)png_get_channels(png_ptr, info_ptr);

    *h = pngHeight;
    *w = pngWidth;
    *depth = channels;
    (*ptr) = std::vector<unsigned char>((*h)*(*w)*(*depth));

    png_bytep *row_pointers =
      (png_bytep*)malloc(sizeof(png_bytep) * channels * (*h));

    unsigned char * ptrArray = &((*ptr)[0]);
    for (int y = 0; y < (*h); ++y)
      row_pointers[y] = (png_byte*) (ptrArray) + rowbytes*y;

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    free(row_pointers);
    return 1;
}

int WritePngStream(FILE * file,
  const vector<unsigned char> & ptr,
  int w,
  int h,
  int depth) {
    png_structp png_ptr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
      return 0;

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
      return 0;

    png_init_io(png_ptr, file);

    // Colour types are defined at png.h:841+.
    char colour;
    switch(depth)
    {
    case 4: colour = PNG_COLOR_TYPE_RGBA;
      break;
    case 3: colour = PNG_COLOR_TYPE_RGB;
      break;
    case 1: colour = PNG_COLOR_TYPE_GRAY;
      break;
    default:
      return 0;
    }

    png_set_IHDR(png_ptr, info_ptr, w, h,
      8, colour, PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    png_bytep *row_pointers =
      (png_bytep*) malloc(sizeof(png_bytep) * depth * h);

    for (int y = 0; y < h; ++y)
      row_pointers[y] = (png_byte*) (&ptr[0]) + w * depth * y;

    png_write_image(png_ptr, row_pointers);
    free(row_pointers);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    return 1;
}
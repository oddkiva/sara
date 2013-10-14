#include "ImageIO.hpp"

FileError::FileError(const std::string& filepath, const std::string& mode)
  : filepath_(filepath), mode_(mode) {}
 
const char* FileError::what() const throw()
{
  std::ostringstream os;
  os 
    << "Error: cannot read file:" << std::endl << filepath_ << std::endl
    << "in mode " << mode_ << std::endl;
  return os.str().c_str();
}

ImageFileHandle::ImageFileHandle(const std::string& filepath,
                                 const std::string& mode)
{
  file_ = fopen(filepath.c_str(), mode.c_str());
  if (!file_)
    throw FileError(filepath, mode);
}

ImageFileHandle::~ImageFileHandle()
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

JpegFileHandle::JpegFileHandle(const std::string& filepath)
  : FileHandle(filepath, "rb")
{
  cinfo_.err = jpeg_std_error(&jerr_.pub);
  jerr_.pub.error_exit = &jpeg_error;

  if (setjmp(jerr_.setjmp_buffer))
  {
    throw FileError(filepath, "rb");
  }

  jpeg_create_decompress(&cinfo_);
  jpeg_stdio_src(&cinfo_, file_);
}

JpegFileHandle::~JpegFileHandle()
{
  jpeg_destroy_decompress(&cinfo_);
}

bool JpegFileHandle::read(unsigned char *& data,
                          int& width, int& height, int& depth)
{
  // Read header file.
  if (!jpeg_read_header(&cinfo_, TRUE))
    return false;
  // Start reading data.
  if (!jpeg_start_decompress(&cinfo_))
    return false;;
  // Read image size.
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

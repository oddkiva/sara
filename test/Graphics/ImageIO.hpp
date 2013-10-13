#include <cstdio>
#include <exception>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

extern "C" {
# include <libjpeg/jpeglib.h>
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
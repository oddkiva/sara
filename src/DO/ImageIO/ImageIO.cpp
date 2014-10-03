// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#if defined(_WIN32) || defined(_WIN32_WCE)
# define NOMINMAX
# include <windows.h>
#endif
#include <DO/ImageIO/ImageIO.hpp>
#include <DO/Core/Image.hpp>
#include <DO/ImageIO/ImageIOObjects.hpp>

using namespace std;

namespace DO {

  static
  inline string fileExtension(const string& filepath)
  {
    string ext( filepath.substr(filepath.find_last_of(".")) );
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
  }

  static
  inline bool isJpegFileExt(const string& ext)
  {
    return 
      ext == ".jpg"  || 
      ext == ".jpeg" || 
      ext == ".jpe"  ||
      ext == ".jfif" ||
      ext == ".jfi";
  }

  static
  inline bool isPngFileExt(const string& ext)
  {
    return ext == ".png";
  }

  static
  inline bool isTiffFileExt(const string& ext)
  {
    return
      ext == ".tif" ||
      ext == ".tiff";
  }

  static
  bool imread(unsigned char *& data, int& w, int& h, int& d,
              const string& filePath)
  {
    data = 0;
    w = h = d = 0;

    string ext(fileExtension(filePath));

    if ( isJpegFileExt(ext) && 
         JpegFileReader(filePath)(data, w, h, d) )
      return true;
    if ( isPngFileExt(ext) &&
         PngFileReader(filePath)(data, w, h, d) )
      return true;
    if ( isTiffFileExt(ext) &&
         TiffFileReader(filePath)(data, w, h, d) )
      return true;

    cerr << "Image format: " << ext << " either currently unsupported or invalid" << endl;
    return false;
  }

  bool readExifInfo(EXIFInfo& exifInfo, const std::string& filePath)
  {
    // Read the JPEG file into a buffer
    FILE *fp = fopen(filePath.c_str(), "rb");
    if (!fp) { 
      cout << "Can't open file:" << endl << filePath << endl; 
      return false; 
    }
    fseek(fp, 0, SEEK_END);
    unsigned long fsize = ftell(fp);
    rewind(fp);
    unsigned char *buf = new unsigned char[fsize];
    if (fread(buf, 1, fsize, fp) != fsize) {
      cout << "Can't read file: " << endl << filePath << endl;
      delete[] buf;
      return false;
    }
    fclose(fp);

    // Parse EXIF
    int code = exifInfo.parseFrom(buf, fsize);
    delete[] buf;
    if (code) {
      //cout << "Error parsing EXIF: code " << code << endl;
      return false;
    }
    return true;
  }

  void print(const EXIFInfo& exifInfo)
  {
    printf("Camera make       : %s\n", exifInfo.Make.c_str());
    printf("Camera model      : %s\n", exifInfo.Model.c_str());
    printf("Software          : %s\n", exifInfo.Software.c_str());
    printf("Bits per sample   : %d\n", exifInfo.BitsPerSample);
    printf("Image width       : %d\n", exifInfo.ImageWidth);
    printf("Image height      : %d\n", exifInfo.ImageHeight);
    printf("Image description : %s\n", exifInfo.ImageDescription.c_str());
    printf("Image orientation : %d\n", exifInfo.Orientation);
    printf("Image copyright   : %s\n", exifInfo.Copyright.c_str());
    printf("Image date/time   : %s\n", exifInfo.DateTime.c_str());
    printf("Original date/time: %s\n", exifInfo.DateTimeOriginal.c_str());
    printf("Digitize date/time: %s\n", exifInfo.DateTimeDigitized.c_str());
    printf("Subsecond time    : %s\n", exifInfo.SubSecTimeOriginal.c_str());
    printf("Exposure time     : 1/%d s\n", (unsigned) (1.0/exifInfo.ExposureTime));
    printf("F-stop            : f/%.1f\n", exifInfo.FNumber);
    printf("ISO speed         : %d\n", exifInfo.ISOSpeedRatings);
    printf("Subject distance  : %f m\n", exifInfo.SubjectDistance);
    printf("Exposure bias     : %f EV\n", exifInfo.ExposureBiasValue);
    printf("Flash used?       : %d\n", exifInfo.Flash);
    printf("Metering mode     : %d\n", exifInfo.MeteringMode);
    printf("Lens focal length : %f mm\n", exifInfo.FocalLength);
    printf("35mm focal length : %u mm\n", exifInfo.FocalLengthIn35mm);
    printf("GPS Latitude      : %f deg (%f deg, %f min, %f sec %c)\n", 
      exifInfo.GeoLocation.Latitude,
      exifInfo.GeoLocation.LatComponents.degrees,
      exifInfo.GeoLocation.LatComponents.minutes,
      exifInfo.GeoLocation.LatComponents.seconds,
      exifInfo.GeoLocation.LatComponents.direction);
    printf("GPS Longitude     : %f deg (%f deg, %f min, %f sec %c)\n", 
      exifInfo.GeoLocation.Longitude,
      exifInfo.GeoLocation.LonComponents.degrees,
      exifInfo.GeoLocation.LonComponents.minutes,
      exifInfo.GeoLocation.LonComponents.seconds,
      exifInfo.GeoLocation.LonComponents.direction);
    printf("GPS Altitude      : %f m\n", exifInfo.GeoLocation.Altitude);
  }

  bool imread(Image<unsigned char>& image, const std::string& filePath)
  {
    unsigned char *data;
    int w, h, d;

    if (!imread(data, w, h, d, filePath))
      return false;

    // Wrap data and get data ownership
    if (d == 1)
      image = Image<unsigned char>(&data[0], Vector2i(w,h), true);
    if (d == 3)
      image = Image<Rgb8>(reinterpret_cast<Rgb8 *>(&data[0]), Vector2i(w,h), true)
        .convert<unsigned char>();
    if (d == 4)
      image = Image<Rgba8>(reinterpret_cast<Rgba8 *>(&data[0]), Vector2i(w,h), true)
        .convert<unsigned char>();

    EXIFInfo info;
    if (readExifInfo(info, filePath))
      flip(image, int(info.Orientation));

    return true;
  }

  bool imread(Image<Rgb8>& image, const std::string& filePath)
  {
    unsigned char *data;
    int w, h, d;

    if (!imread(data, w, h, d, filePath))
      return false;

    // Wrap data and get data ownership
    if (d == 1) {
      image = Image<unsigned char>(&data[0], Vector2i(w,h), true)
        .convert<Rgb8>();
    } else if (d == 3) {
      image = Image<Rgb8>(reinterpret_cast<Rgb8 *>(&data[0]), Vector2i(w,h), true);
    } else if (d == 4) {
      image = Image<Rgba8>(reinterpret_cast<Rgba8 *>(&data[0]), Vector2i(w,h), true)
        .convert<Rgb8>();
    }

    EXIFInfo info;
    if (readExifInfo(info, filePath))
      flip(image, int(info.Orientation));

    return true;
  }

  bool saveJpeg(const Image<Rgb8>& image, const std::string& filePath,
                int quality)
  {
    JpegFileWriter jpegWriter(
      reinterpret_cast<const unsigned char *>(image.data()),
      image.width(), image.height(), 3);
    return jpegWriter(filePath, quality);
  }

} /* namespace DO */
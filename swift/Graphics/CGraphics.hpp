#pragma once

// Image I/O.
void* JpegImageReader_init(const char* name);
void JpegImageReader_deinit(void* reader);
void JpegImageReader_imageSizes(void* reader, int* w, int* h, int* c);
void JpegImageReader_readImageData(void* reader, unsigned char* dataPtr);

// Video I/O.
void* VideoStream_init(const char* name);
void VideoStream_deinit(void* stream);
unsigned char* VideoStream_getFramePtr(void* stream);
int VideoStream_getFrameWidth(void* stream);
int VideoStream_getFrameHeight(void* stream);
int VideoStream_readFrame(void* stream);

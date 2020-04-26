/**
 *  @file app_recursive.cc
 *  @brief Example of an application of recursive filtering in the CPU and the GPU
 *  @author Diego Nehab
 *  @author Andre Maximo
 *  @date November, 2011
 */

#include <cstdio>
#include <cstddef>
#include <cstdarg>
#include <cstdlib>

#include "cv.h"
#include "highgui.h"

#include <gpufilter.h>
#include <cpuground.h>

typedef unsigned char uchar;

//-----------------------------------------------------------------------------
// Aborts and prints an error message
void errorf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(1);
}

//-----------------------------------------------------------------------------
// Runs a given filter
int main(int argc, char *argv[]) {
    if( argc < 3 )
        errorf(
            "Usage:\n"
            "  recursive [options] <input image> <output image>\n"
            "where options are:\n"
            "  -unit:<cpu|gpu>\n"
            "    default: cpu\n"
            "  -filter:<gaussian|bspline3i>\n"
            "    default: gaussian\n"
            "  -sigma:<value> (for Gaussian)\n"
            "    default: 1.0"
            );

    // process options
    const char *filter = "gaussian";
    const char *file_in = 0, *file_out = 0;
    const char *unit = "cpu";

    float sigma = 1.f;
    int end = 0;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-filter:", sizeof("-filter:")-1) == 0) {
            filter = argv[i]+sizeof("-filter:")-1;
        } else if (strncmp(argv[i], "-unit:", sizeof("-unit:")-1) == 0) {
            unit = argv[i]+sizeof("-unit:")-1;
        } else if (sscanf(argv[i], "-sigma:%f%n", &sigma, &end) == 1) {
            if (argv[i][end] != '\0' || sigma < 0.f)
                errorf("Invalid argument '%s'", argv[i]);
        } else if (argv[i][0] == '-') {
            errorf("Unknown option '%s'", argv[i]);
        } else {
            if (!file_in) file_in = argv[i];
            else if (!file_out) file_out = argv[i];
            else errorf("Too many arguments");
        }
    }

    if( !file_in || !file_out )
        errorf("Not enough arguments!");

    printf("Loading input image '%s'\n", file_in);

    IplImage *in_img = cvLoadImage(file_in, CV_LOAD_IMAGE_UNCHANGED);

    if( !in_img )
        errorf("Unable to load image '%s'", file_in);

    int in_w = in_img->width, in_h = in_img->height, depth = in_img->nChannels;

    printf("Image is %dx%dx%d\n", in_w, in_h, depth);

    printf("Flattening input image\n");

    float **flat_in = new float*[depth];

    for (int c = 0; c < depth; c++)
        flat_in[c] = new float[in_w*in_h];

    if( !flat_in ) 
        errorf("Out of memory!");

    for (int c = 0; c < depth; c++) {

        cvSetImageCOI(in_img, c+1);

        IplImage *ch_img = cvCreateImage(cvSize(in_w, in_h), in_img->depth, 1);

        cvCopy(in_img, ch_img);

        IplImage *uc_img = cvCreateImage(cvSize(in_w, in_h), IPL_DEPTH_8U, 1);

        cvConvertImage(ch_img, uc_img);

        for (int i = 0; i < in_h; ++i)
            for (int j = 0; j < in_w; ++j)
                flat_in[c][i*in_w+j] =
                    ((uchar*)(uc_img->imageData+i*uc_img->widthStep))[j]/255.f;

        cvReleaseImage(&ch_img);
        cvReleaseImage(&uc_img);

    }

    if( strcmp(filter, "gaussian") == 0 ) {

        printf("Applying filter gaussian in the %s (sigma = %g)\n",
               unit, sigma);

        if( strcmp(unit, "cpu") == 0 )
            gpufilter::gaussian_cpu(flat_in, in_w, in_h, depth, sigma);
        else if( strcmp(unit, "gpu") == 0 )
            gpufilter::gaussian_gpu(flat_in, in_h, in_w, depth, sigma);
        else
            printf("Nothing done: invalid unit!\n");

    } else if (strcmp(filter, "bspline3i") == 0) {

        printf("Applying filter bspline3i in the %s\n", unit);

        if( strcmp(unit, "cpu") == 0 )
            gpufilter::bspline3i_cpu(flat_in, in_w, in_h, depth);
        else if( strcmp(unit, "gpu") == 0 )
            gpufilter::bspline3i_gpu(flat_in, in_h, in_w, depth);
        else
            printf("Nothing done: invalid unit!\n");

    } else {

        errorf("Unknown method '%s'", filter);

    }

    printf("Packing output image\n");

    IplImage *out_img = cvCreateImage(cvSize(in_w, in_h), IPL_DEPTH_8U, depth);

    gpufilter::_clamp clamp;
    for (int c = 0; c < depth; c++)
        for (int i = 0; i < in_h; ++i)
            for (int j = 0; j < in_w; ++j)
                ((uchar*)(out_img->imageData+i*out_img->widthStep))
                    [j*depth + c] = (uchar)(floorf(clamp(flat_in[c][i*in_w+j])
                                                   *255.f+0.5f));

    printf("Saving output image '%s'\n", file_out);

    if( !cvSaveImage(file_out, out_img) )
        errorf("Could not save: %s\n", file_out);

    cvReleaseImage(&in_img);
    cvReleaseImage(&out_img);

    return 0;
}

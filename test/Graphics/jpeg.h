#include <stdio.h>
#include "jpeglib.h"
#include <setjmp.h>


extern JSAMPLE * image_buffer;	/* Points to large array of R,G,B-order data */
extern int image_height;	/* Number of rows in image */
extern int image_width;		/* Number of columns in image */

struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */
  jmp_buf setjmp_buffer;	/* for return to caller */
};

typedef struct my_error_mgr * my_error_ptr;

METHODDEF(void) my_error_exit (j_common_ptr cinfo);

GLOBAL(int) read_JPEG_file (char * filename);
GLOBAL(void) write_JPEG_file (char * filename, int quality);
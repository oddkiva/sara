#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#  include <OpenCL/cl.h>
#else
#  include <CL/cl.h>
#endif

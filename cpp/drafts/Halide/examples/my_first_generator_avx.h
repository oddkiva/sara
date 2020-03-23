#ifndef HALIDE_____my_first_generator_avx_h
#define HALIDE_____my_first_generator_avx_h
#include <stdint.h>

// Forward declarations of the types used in the interface
// to the Halide pipeline.
//
// For the definitions of these structs, include HalideRuntime.h

// Halide's representation of a multi-dimensional array.
// Halide::Runtime::Buffer is a more user-friendly wrapper
// around this. Its declaration is in HalideBuffer.h
struct halide_buffer_t;

// Metadata describing the arguments to the generated function.
// Used to construct calls to the _argv version of the function.
struct halide_filter_metadata_t;

#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif



#ifdef __cplusplus
extern "C" {
#endif

int my_first_generator_avx(uint8_t _offset, struct halide_buffer_t *_input_buffer, struct halide_buffer_t *_brighter_buffer) HALIDE_FUNCTION_ATTRS;
int my_first_generator_avx_argv(void **args) HALIDE_FUNCTION_ATTRS;
const struct halide_filter_metadata_t *my_first_generator_avx_metadata() HALIDE_FUNCTION_ATTRS;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

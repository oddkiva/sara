1. Use texture or layered texture/surface for the image pyramid.
2. Atomic addition of the gradient histogram.
3. Atomic add operation for SIFT.

// Send GPU operations asynchronously... No need since everything is
// sequential for Gaussian pyramid.
//
// From then on, yes everything can be asynchronous:
// - DoG pyramids
// - Extremum map pyramids
// - Gradient pyramids
//
// cudaStream_t stream = 0;
// cudaStreamCreate(&stream);
// cudaStreamSynchronize(stream);
// cudaStreamDestroy(stream);

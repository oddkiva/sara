TODO
====

The Halide implementation of SIFT has been written with good confidence regarding the
implementation details:

Integration
-----------
- [x] Integrate the peak localization
- [x] Integrate the peak refinement


Unit Tests
----------
- [x] Orientation histogram
- [x] Repeated box blur filters
- [x] Orientation peak localization
- [x] Orientation peak residual estimation
- [x] SIFT descriptor
      Compare with the initial CPU implementation (without normalization).
- [x] SIFT descriptor normalization

Optimization
------------
- [ ] Minimize transfers back-and-forth between host memory and device memory.
      - [ ] Provide API in which only we interact only with
            `Halide::Runtime::Buffer<T>` objects.

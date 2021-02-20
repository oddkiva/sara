OPENGL
======
- Learn how to use framebuffer for texture streaming.

VULKAN
======
- Hello Triangle tutorial.

HALIDE
======

The Halide implementation of SIFT has been written with good confidence regarding the
implementation details:

Optimization
------------
- [ ] Minimize transfers back-and-forth between host memory and device memory.
      - [ ] Provide API in which only we interact only with
            `Halide::Runtime::Buffer<T>` objects.

Optical Distortion
==================
Implement the different inverse distortion method.
- power series approach
- bisection method


Autocalibration with radial distortion
======================================
Reimplement Fitzgibbon's approach (fundamental matrix with one radial distortion
coefficients).

Bundle Adjustment Tests
=======================

Form the bundle adjustment problem.
TODO:
- Choose a reference camera.
- Populate the list of cameras in BundleAdjustmentProblems:
  As a reminder:
  image #i -> camera #i -> BundleAdjustmentProblem.camera_parameters.row(i)
- Initialize the absolute camera poses.

Vanishing Point Detection with Radial Distortion
================================================
Unsupervised Vanishing Point Detection and Camera Calibration from a Single
Manhattan Image with Radial Distortion

- Plucker coordinates for 3D line computations.
- LCC = line of circle center, bisector of the chord.
- Circle fitting: second moment matrix, scattering ellipse
- centroid of the chord determined from the scattering ellipse
- angle between the LCC and the x-axis is determined.

Level Sets
==========
- fix the fast marching with dealing with negative sign.

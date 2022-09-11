VULKAN
======
- Hello Triangle tutorial.


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

Camera Resectioning Tests
=========================

1. Enumerate the vertex coordinates of a cube in the world coordinates.
   *DONE*

2. Create one normalized camera that displaced from the world center.
   *DONE*

3. Project the cube vertices to the images.
   *DONE*


Bundle Adjustment Tests
=======================

4. Form the bundle adjustment problem.
   TODO:
   - Choose a reference camera.
   - Populate the list of cameras in BundleAdjustmentProblems:
     As a reminder:
     image #i -> camera #i -> BundleAdjustmentProblem.camera_parameters.row(i)
   - Initialize the absolute camera poses.

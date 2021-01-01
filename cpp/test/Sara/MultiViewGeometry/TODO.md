Bundle Adjustment Tests
=======================


Reminder
========
The form for the essential matrix is E = [t]x R
where:
- `R` is the rotation.
- `t` is the translation
The transformation (R, t) is the relative movement from the initial camera to
the second camera:
- the initial camera has moved to the second camera position by a displacement `t`.
- the initial camera gaze direction has rotated to the second camera gaze
  direction by a quantity `R`.



1. Create cube data.
   Easy but lazy

2. Create 3 to 4 camera normalized cameras
   Easy but lazy
   Generate absolute position.

3. Calculate the relative poses.
   Easy but refresh memory.

4. Project the cube vertices to the images.
   Easy but need to refresh memory

5. Form the bundle adjustment problem.
   TODO:
   - Choose a reference camera.
   - Populate the list of cameras in BundleAdjustmentProblems:
     As a reminder:
     image #i -> camera #i -> BundleAdjustmentProblem.camera_parameters.row(i)
   - Initialize the absolute camera poses.

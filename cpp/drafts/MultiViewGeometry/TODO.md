Vanishing Point Detection
=========================
- Barnard's Gaussian sphere:
  - interpretation line
  - vp = pole, dual of point = equator
  - n = (u1 x u2) x (v1 x v2)  (normalized coordinates).
- 3-line RANSAC: Bazin
- Rother approach: two nice distance functions:
  - distance(line, line segment), distance(line segment, vp)

Unsupervised Vanishing Point Detection and Camera Calibrationfrom a Single Manhattan Image with Radial Distortion
=================================================================================================================
- Plucker coordinates for 3D line computations.
- LCC = line of circle center, bisector of the chord.
- Circle fitting: second moment matrix, scattering ellipse
- centroid of the chord determined from the scattering ellipse
- angle between the LCC and the x-axis is determined.

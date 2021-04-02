Understanding the raw data in KITTI
===================================

KITTI is a very comprehensive benchmark suite that touches upon many relevant
computer vision tasks. This benchmark is clearly geared towards road safety
applications.

In this section, I provide some more technical notes regarding the available
data and which I find worth emphasizing again. The raw data in the benchmark is
very large and it can be quite confusing to find our way as the documentation is
scattered in many places across the website.

*N.B.: these notes reflect the very best of my current understanding. They may
still contain errors and I am very interested to know about them if you spot any
misunderstandings on my part and explain them to me.*


Object Detection Challenge
--------------------------

In the object detection challenge -- be it whether 2D or 3D, it does not matter
since the provided data provided is pretty much the same for both tasks -- it is
worth focusing our attention on deciphering the calibration data.

Camera Sensors
**************

As depicted in the images from the setup page
(http://www.cvlibs.net/datasets/kitti/setup.php), the car is equipped with:

- 4 cameras
- 1 Velodyne lidar scanner
- 1 IMU/GPS device.

Each camera is at a height of :math:`1.65\text{m}` above the ground plane.
As highlighted in the images, the camera coordinate system follows OpenCV's
camera axis convention where:

- The :math:`x`-axis is the horizontal axis pointing to the right of the vehicle.
- The :math:`y`-axis is the vertical axis pointing down to the ground.
- The :math:`z`-axis is the axis pointing from the camera center to the camera gaze direction.

3D Geometry of Annotations
**************************

For a given image, we are provided a label file that lists object annotations
where each object annotation contains the following geometry:

- A 2D bounding box in the image (streamed from Camera #2) with 2D image
  coordinates:

  - `left`
  - `top`
  - `right`
  - `bottom`

  These coordinates are straightforward to understand. Now let us move onto the
  description of the 3D bounding box which we describe with more details.

- A 3D oriented bounding box with:

  - one 3D reference point expressed in the camera coordinate system
    :math:`\mathbf{t} = (x, y, z)`.

    Visually this corresponds to the center of the bottom face of the 3D
    bounding box, i.e, the face lying on the ground surface.

  - the 3D dimensions of the bounding box :math:`(w, h, l)`. This allows us to
    populate the :math:`8` vertices in the **local object coordinate system**:

    .. math::

       \mathbf{X}_l =
       \left[
       \begin{array}{r|r|r|r|r|r|r|r}
       -w/2 &  w/2  &  w/2 & -w/2 & -w/2 &  w/2  & w/2 & -w/2 \\
          0 &    0  &   -h &   -h &    0 &    0  &  -h &   -h \\
       -l/2 & -l/2  & -l/2 & -l/2 &  l/2 &  l/2  & l/2 &  l/2 \\
          1 &    1  &    1 &    1  &   1 &    1  &   1 &    1 \\
       \end{array}
       \right]

    Upon reading the coordinates:

    - The first 4 columns are the vertices of the front-face of the bounding
      box.
    - The last 4 columns are the vertices of the back-face of the bounding
      box.

    *N.B.: it is possible that I permutated some variables.*

    In this local object coordinate system, the origin is at the center of the
    bottom face.

  - the yaw angle :math:`\psi` of the bounding box with respect to the
    :math:`y`-axis of the camera coordinate system

    .. math::

       \mathbf{R}_\psi =
       \begin{bmatrix}
        \cos\psi &  0  & \sin\psi \\
               0 &  1  &        0 \\
       -\sin\psi &  0  & \cos\psi \\
       \end{bmatrix}

It follows that the vertices expressed in the camera coordinate system can be
calculated as

.. math::

   \mathbf{X} =
   \begin{bmatrix}
   \mathbf{R}_\psi & \mathbf{t} \\
    \mathbf{0}_3^T &          1 \\
   \end{bmatrix}
   \mathbf{X}_l

We can double-check these by examining the MATLAB code provided in the
development kit.


Projection of 3D World Points to the Camera #2's Film
*****************************************************

One important thing to note is that the camera coordinate system refers to the
coordinate system associated to Camera #0. Thus the vertices :math:`\mathbf{X}`
are expressed with respect to Camera #0's coordinate system and not w.r.t. to
Camera #2's coordinate system!

Indeed the **README** file in the development kit states that the projection of
the 3D bounding box to the image is done via the following formula

.. math::

   \mathbf{u} = \mathbf{P}_2
     \left[
       \begin{array}{c|c}
       \mathbf{R}_0^{(\text{rect})} & \mathbf{0}_3 \\
       \hline
                     \mathbf{0}_3^T &            1 \\
       \end{array}
     \right]
     \mathbf{X}

where

- :math:`\mathbf{u}` is the 2D pixel coordinates on camera #2,
- :math:`\mathbf{X}` are the 3D vertices of the bounding box in
  the camera coordinate system which we described in details in the subsection
  above.

Splitting the projection matrix as

.. math::

   \mathbf{P}_2 =
     \left[
     \begin{array}{c|c}
       \mathbf{M} & \mathbf{m}
     \end{array}
     \right]

It appears upon examination of the calibration data (see for example text file
**0000.txt**) that the projection matrix :math:`\mathbf{P}_2` can be decomposed
simply as:

.. math::

   \mathbf{P}_2 = \mathbf{K}
     \left[
     \begin{array}{c|c}
       \mathbf{I}_3 & \mathbf{t}
     \end{array}
     \right]

where:

- :math:`\mathbf{K} = \mathbf{M}` is the usual calibration matrix.
- :math:`\mathbf{t} = \mathbf{K}^{-1} \mathbf{m}` relates the metric
  displacement of camera center #2 w.r.t. camera center #0.

The interpretation we can make from this decomposition is that as a first
approximation, cameras #0, #1, #2 and #3 have their axes exactly aligned.

This is not the case obviously and the rotation matrix
:math:`\mathbf{R}_0^{(\text{rect})}` quantifies the small angular differences
between cameras #0 and #2.

Now let us reinject the decomposition of :math:`\mathbf{P}_2` in the projection
equation

.. math::

   \mathbf{u} = \mathbf{K}
     \left[
       \begin{array}{c|c}
       \mathbf{I}_3 & \mathbf{t} \\
       \end{array}
     \right]

     \left[
       \begin{array}{c|c}
       \mathbf{R}_0^{(\text{rect})} & \mathbf{0}_3 \\
       \hline
                     \mathbf{0}_3^T &            1 \\
       \end{array}
     \right]
     \mathbf{X} \\


Then multiplying the matrix blocks,

.. math::

   \mathbf{u} = \mathbf{K}
     \left[
       \begin{array}{c|c}
       \mathbf{I}_3\ \mathbf{R}_0^{(\text{rect})} + \mathbf{t}\ \mathbf{0}_3^T &
       \mathbf{I}_3\ \mathbf{0}_3 + \mathbf{t}\ 1  \\
       \end{array}
     \right]

     \mathbf{X} \\

Upon simplification,

.. math::

   \mathbf{u} = \mathbf{K}
     \left[
       \begin{array}{c|c}
       \mathbf{R}_0^{(\text{rect})} & \mathbf{t} \\
       \end{array}
     \right]
     \mathbf{X}.

Now the equation has the familiar form as exposed in usual computer vision
textbooks. And it follows from this equation that to go from camera coordinate
system #0 to camera coordinate system #2 is done by the rigid body motion
:math:`(\mathbf{R}_0^{(\text{rect})}, \mathbf{t})`.

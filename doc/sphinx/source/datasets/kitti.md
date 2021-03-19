Understanding the raw data in KITTI
===================================

KITTI is a very comprehensive benchmark suite that touches upon many relevant
computer vision tasks. It is strongly geared towards autonomous driving
applications.

The raw data in the benchmark is very large and it can be quite confusing to
find our way as the documentation is scattered in many places.

*N.B.: these notes reflect the very best of my current understanding. They may
still contain errors and I am very interested to know about them if you spot any
misunderstandings on my part and explain to me.*


Object Detection Challenge
--------------------------

In the object detection challenge, be it whether 2D or 3D -- it does not matter
since the data provided is pretty much the same, it is worth understanding what
the calibration data is about.

As depicted in the images from the setup page
(http://www.cvlibs.net/datasets/kitti/setup.php), the car is equipped with:

- 4 cameras
- 1 Velodyne lidar scanner
- 1 IMU/GPS device.

Each camera is at a height of :math:`1.65\text{m}` above the ground plane.

For a given image, we are provided a label file that lists of object
annotations. Each object annotation contains:

- A 2D bounding box in images which are imaged from Camera #2.
- A 3D oriented bounding box with:

  - one 3D reference point expressed in the camera coordinate system, which
    corresponds to a point on the ground plane :math:`(x, y, z)`.
  - the 3D dimensions of the bounding box :math:`(h, w, l)`.
  - the yaw angle :math:`\psi` of the bounding box with respect to the
    :math:`y`-axis of the camera coordinate system

Now the camera coordinate system refers to the coordinate system associated to
Camera #0.

The **README** file found in the development kit archive states that to project
the 3D bounding box to the image is obtained from the formula:

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

Upon examining the calibration data, it turns out that the projection matrix
:math:`\mathbf{P}_2` can be decomposed in practice as:

.. math::

   \mathbf{P}_2 = \mathbf{K}
     \left[
     \begin{array}{c|c}
       \mathbf{I}_3 & \mathbf{t}
     \end{array}
     \right]

The interpretation we can make is that as a first approximation, each camera
have their axes exactly aligned.

In practice, this is not the case and the rotation matrix
:math:`\mathbf{R}_0^{(\text{rect})}` provides the small angular differences
between cameras #0 and #2.

So rewriting the projection equation as

.. math::

   \mathbf{u} = \mathbf{K}_2
     \left[
       \begin{array}{c|c}
       \mathbf{R}_0^{(\text{rect})} & \mathbf{t} \\
       \hline
                     \mathbf{0}_3^T &          1 \\
       \end{array}
     \right]
     \mathbf{X}

It follows from this equation that to go from camera coordinate system #0 to
camera coordinate system #2 is done by the rigid body transform
:math:`(\mathbf{R}_0^{(\text{rect})}, \mathbf{t})`.

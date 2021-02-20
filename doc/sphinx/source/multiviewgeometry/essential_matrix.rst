.. _chap-essential-matrix:

Essential Matrix and Relative Motion
====================================

Consider two cameras :math:`C_0` and :math:`C_1`. In the sequel, we can sort of
imagine that:

- the coordinate system of the first camera :math:`C_0` is the world coordinate
  system.
- the coordinate system of the second camera :math:`C_1` is a convenient local
  coordinate system.

Consider a 3D point with world coordinate system :math:`\mathbf{X}^0` and
suppose we can calculate its coordinates :math:`\mathbf{X}^1` in this local
coordinate system with the following rigid body motion

.. math::

   \mathbf{X}^1 = \mathbf{R} \mathbf{X}^0 + \mathbf{t} .

We can interpret the formula as follows:

- The column vectors of the rotation matrix :math:`\mathbf{R}` are the
  coordinates of the world axes expressed in the local coordinate system.
- :math:`\mathbf{t}` are the coordinates of the world origin expressed in the
  local camera coordinate system.

Basically this rigid body describes the relative position of camera :math:`C_0`
w.r.t. camera :math:`C_1` (and in its local coordinate system), such that:

- the local camera axes are rotated to the world camera axes by the rotation
  :math:`\mathbf{R}`.
- the origin of local coordinate system is moved to the world origin position by
  the translation :math:`\mathbf{t}`.

The essential matrix we implement in *Sara* uses *this* rotation matrix and
*this* translation vector.

.. math::

   \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}\ .

The relative *motion* extracted from the essential matrix of camera
:math:`C_1` w.r.t. :math:`C_0` will mean this rotation and translation pair
:math:`(\mathbf{R}, \mathbf{t})`. And be careful, this is not the same
convention described in the wikipedia page.


Now when I talk about the relative **pose** of camera :math:`C_1` w.r.t. to camera
:math:`C_0`, I will mean the inverse rigid body motion
:math:`(\mathbf{R}_{0 \rightarrow 1}, \mathbf{t}_{0 \rightarrow 1})`.

The reverse rigid body motion is obtained from:

.. math::

   \begin{aligned}

   \mathbf{R} \mathbf{X}^0 + \mathbf{t} &= \mathbf{X}^1 \\

   \mathbf{X}^0 + \mathbf{R^T} \mathbf{t} &= \mathbf{R}^T \mathbf{X}^1 \\

   \mathbf{X}^0 &= \mathbf{R}^T \mathbf{X}^1 - \mathbf{R^T} \mathbf{t} \\

   \end{aligned}

By interpreting the inverse rigid body motion, we see that:

- the position of camera :math:`C_1` w.r.t. camera coordinate system :math:`C_0`,
  is calculated as :math:`\mathbf{t}_{0 \rightarrow 1} = -\mathbf{R}^T \mathbf{t}`.
- the "gaze orientation", i.e. the rotation matrix, of camera :math:`C_1` w.r.t. camera
  coordinate system :math:`C_0` is calculated as :math:`\mathbf{R}_{0
  \rightarrow 1} = \mathbf{R}^T`

Quick Summary
-------------
The following remarks are useful to debug code dealing with structure-from-motion:

- The coordinates of the first camera center in the first camera coordinate
  system is :math:`(0, 0, 0)`.
- The coordinates of the first camera center in the second camera coordinate
  system is :math:`\mathbf{t}`.
- The coordinates of the second camera center in the second camera coordinate
  system is :math:`(0, 0, 0)`.
- The coordinates of the second camera center in the first camera coordinate
  system is :math:`-\mathbf{R}^T \mathbf{t}`.

- :math:`\mathbf{t}` goes from camera :math:`C_1` to camera :math:`C_0` and not
  the other way around.


Bundle Adjustment
=================
In the bundle adjustment problem we need to initialize the position and the gaze
direction of each camera :math:`C_i = (\mathbf{R}_i, \mathbf{t}_i)` in a global
coordinate system. The position and the gaze direction of the camera is called
the camera pose.

To start, we choose one camera, say :math:`C_0`, and its associated camera
coordinate system will be set as the world coordinate system.

The relative pose of camera :math:`C_j` w.r.t. camera :math:`C_i` is the
rotation and translation pair :math:`(\mathbf{R}_{i \rightarrow j},
\mathbf{t}_{i \rightarrow j})`. Let us now shorten the notations by writing them
as :math:`(\mathbf{R}_{ij}, \mathbf{t}_{ij})` The relative motion is recovered
from the estimation of the essential matrix :math:`\mathbf{E}_{ij}`.

Let us assume the camera network is fully connected from now on. To retrieve
the absolute pose of each camera :math:`C_i`, we can retrieve the shortest path
of connected cameras :math:`C_0, C_{i_1}, C_{i_2},\dots, C_{i}`. Composing
successively the relative motions, we would retrieve the absolute pose by
applying recursively:

.. math::
   \mathbf{R}_{j} = \mathbf{R}_{i} \mathbf{R}_{ij} \mathbf{R}_{i}^T \\

.. math::
   \mathbf{t}_j = \mathbf{t}_{ij} + \mathbf{t}_i

Because the relative pose estimations are noisy, the successive composition of
rotation and translation will induce an accumulation of errors. And as a
consequence the estimated global camera pose can drift very far from the ground
truth pose.

Unfortunately, we cannot add translations together like this because the
relative pose from the essential matrix allows us to recover the translation
only up to a scale. In other words, we can only know the direction vector of the
translation but not the magnitude of the translation. Thus it is not a good
approach.

Fortunately, as described in Snavely's paper, we can do like Bundler. Bundler
starts from two cameras. From these two cameras, we can initialize the 3D
geometry up to a scale by estimating the essential matrix and then by
triangulation. A bundle adjustment is then applied to refine the 3D geometry and
the camera parameters.

Then a third new camera is added if its view overlaps with at least one of the
two cameras. Because the 3D geometry is initialized, we know the correspondences
between the 3D world points and the 2D image points in each views. Since the 3D
image points are also imaged in the third view, we can initialize the third
camera pose by camera resectioning.

The internal camera parameters are initialized from the extraction of EXIF
metadata. Again a bundle adjustment is then performed done on all the three
views to refine the camera parameters and 3D geometry.

By proceeding incrementally in this manner, where each time a new camera is
added, its pose is initialized by camera resectioning and a bundle adjustment is
performed.

In the end the epipolar geometry serves mostly to:

- initialize the 3D geometry from the seed two-view geometry.
- track the correspondence between the 3D world points and the 2D image points.

In the next section we describe the DLT method to initialize the pose.


Camera Resectioning for Incremental Bundle Adjustment
=====================================================

The relative pose estimation allows to recover the position of two cameras. How
do we choose the third camera and initialize its pose?

To start, we can look for the third image where the 3D points estimated from the
two-view geometry reappears the most.

The DLT initializes the camera pose and the internal parameters are initialized
from EXIF metadata. What the DLT solves is also called **camera resectioning**.

In summary the data we know are:

- the internal camera matrix :math:`\mathbf{K}`
- the image coordinates in the third image :math:`\mathbf{x}_i`
- the normalized camera coordinates :math:`\tilde{\mathbf{x}}_i = \mathbf{K}^{-1} \mathbf{x}_i`
- the 3D points are calculated from the relative pose :math:`\mathbf{X}_i`

We want to determine the pose of the third camera, i.e.:

- the global rotation :math:`\mathbf{R}`
- the global translation :math:`\mathbf{t}`

Projecting the 3D points to the image:

.. math::
   \mathbf{x}_i = \mathbf{K} [\mathbf{R} | \mathbf{t}] \mathbf{X}_i \\

   \tilde{\mathbf{x}}_i = \mathbf{R} \mathbf{X}_i + \mathbf{t} \\

The third image needs to have at least :math:`n \geq 6` point correspondences
between 3D points :math:`\mathbf{X}_i` and 2D image points :math:`\mathbf{x}_i`
to fully retrieve the third camera pose.

We are then able to form a new bundle adjustment problem involving the three
cameras to refine again the 3D points and camera parameters (both external and
internal).

It turns out in fact that we may not even need to know the camera internal
parameters. [Hartley and Zisserman] calculate the whole projection matrix

.. math::
   \mathbf{P} = \mathbf{K} [\mathbf{R} | \mathbf{t}]

and then decompose the matrix :math:`\mathbf{P}` to fully recover the internal
and external parameters. More accurate details in:
https://users.cecs.anu.edu.au/~hartley/Papers/CVPR99-tutorial/tutorial.pdf

Proceeding incrementally like this, we can also retrieve the next camera poses.

The DLT approach is in theory only applicable to the pinhole camera model. It
can be a good initialization for the bundle adjustment which will estimate the
distortion coefficients of the camera.

References
----------
The DLT was proposed by [Hartley and Zisserman 1999] and is the simplest one to
implement.

More robust approaches are proposed later:

- Lepetit et al.'s EPnP approach (IJCV 2008) which is better.
- Lambda-twist

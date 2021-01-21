.. _chap-essential-matrix:

Essential Matrix and Relative Motion
====================================

Consider two cameras :math:`C_0` and :math:`C_1`. In this setup, the coordinate
system of the first camera :math:`C_0` is considered to be the world coordinate
system.

Let us denote a 3D point expressed in the world coordinate system by
:math:`\mathbf{X}^0`. We can calculate its coordinates :math:`\mathbf{X}^1` in
the second coordinate system by the rigid body motion

.. math::

   \mathbf{X}^1 = \mathbf{R} \mathbf{X}^0 + \mathbf{t} .

We also denote the 3D vector expressed in the first coordinate system by
:math:`\mathbf{d}^0` . We can calculate its coordinates in the second coordinate
system as

.. math::

   \mathbf{d}^1 = \mathbf{R} \mathbf{d}^0.

Thus the essential matrix we implement in *Sara* has the following form

.. math::

   \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}\ .

Be careful: this is not the same convention described in the wikipedia page.

One way of interpreting the relative motion is as follows.

- The columns of :math:`\mathbf{R}` are the coordinates of the world axes w.r.t.
  the camera coordinate system.
- The vector :math:`\mathbf{t}` are the coordinates of the world center w.r.t.
  the camera coordinate system.

Rephrasing again, the relative motion :math:`(\mathbf{R}, \mathbf{t})` can be
understood as follows.

- The first camera center wrt to the second camera center by a translational
  quantity :math:`-\mathbf{t}`. Notice here the negative sign.
- The initial camera gaze direction has moved to the second camera gaze
  direction by a rotational quantity :math:`\mathbf{R}^T`.
- The coordinates of these displacements are expressed w.r.t. the second camera
  coordinates.

Remarks
-------
The following remarks are useful for debugging purposes:

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

To start, we choose one camera, say :math:`C_0` and its associated camera
coordinate system will be set as the world coordinate system.

The relative motion **from** camera :math:`C_i` **to** camera :math:`C_j` is the
rotation and translation pair :math:`(\mathbf{R}_{ij}, \mathbf{t}_{ij})`.  The
relative motion is recovered from the estimation of the essential matrix
:math:`\mathbf{E}_{ij}`.

Let us assume the camera network is fully connected from now on. To retrieve
the absolute pose of each camera :math:`C_i`, we can retrieve the shortest path
of connected cameras :math:`C_0, C_{i_1}, C_{i_2},\dots, C_i`. Composing
successively the relative motion, we retrieve the global pose:

.. math::
   \mathbf{R}_i = \mathbf{R}_{ji} \dots \mathbf{R}_{i_1 i_2} \mathbf{R}_{0 i_1} \\

   \mathbf{t}_i = \mathbf{t}_{ji} + \dots + \mathbf{t}_{i_1 i_2} + \mathbf{t}_{0 i_1}


Because the relative pose estimations are noisy, the successive composition of
rotation and translation will induce an accumulation of errors.

However the addition of translations is the most problematic because the
relative pose estimation can only recover the translation up to a scale. In
other words, it can only recover the direction of the translation.

Thus it is not a good approach. Instead as described in Snavely's paper, Bundler
starts from two cameras. Then a new camera is added and its pose is initialized
with the Direct Linear Transform (DLT). The internal camera parameters are
initialized from the extraction of EXIF metadata.

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

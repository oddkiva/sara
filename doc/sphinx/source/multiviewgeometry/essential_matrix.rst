.. _chap-essential-matrix:

Essential Matrix and Relative Motion
====================================

The essential matrix we implement in Sara has the following form

.. math::

   \mathbf{E} = [\mathbf{t}]_\times \mathbf{R}.

where:

- :math:`\mathbf{R} \in \mathbf{SO}(3)` is the relative rotation from the *first* camera to the
  *second* camera.
- :math:`\mathbf{t} \in \mathbf{R}^3` is the relative translation from the
  *first* camera to the *second* camera.

The coordinates of the :math:`\mathbf{t}` is expressed with respect to the
**first** camera coordinates system.

Rephrasing again, the relative motion :math:`(\mathbf{R}, \mathbf{t})` can be understood
as follows.

- The initial camera has moved from to the second camera position by a
  translational quantity :math:`\mathbf{t}`.
- The initial camera gaze direction has moved to the second camera gaze
  direction by a rotational quantity :math:`\mathbf{R}`.


Bundle Adjustment
=================
In the bundle adjustment problem we need to initialize the position and the gaze
direction of each camera :math:`C_i = (\mathbf{R}_i, \mathbf{t}_i)` in a global
coordinate system. The position and the gaze direction of the camera is called
the camera pose.

To start, we choose one camera, say :math:`C_0` and its associated camera
coordinate system will be set as the the world coordinate.

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
starts from two cameras. Then a new camera is added, its pose is initialized
with the direct linear transform. The internal camera parameters are initialized
from the extraction of EXIF tags.


Direct Linear Transform for Incremental Bundle Adjustment
=========================================================

Referenced as [Hartley and Zisserman 2004].

The relative pose estimation allows to recover the position of two cameras. How
do we choose the third camera and initialize its pose? To do so, we can look the
third image for which the image correspondences reappears the most

As said early, the direct linear transform aims at recovering the camera pose
and the internal camera parameters :math:`\mathbf{K}` are extracted from EXIF
tags.

In summary the data we know are:

- the internal camera matrix :math:`\mathbf{K}`
- the image coordinates in the third image :math:`\mathbf{x}_i`
- the normalized camera coordinates :math:`\tilde{\mathbf{x}}_i = \mathbf{K}^{-1} \mathbf{x}_i`
- the 3D points are calculated from the relative pose :math:`\mathbf{X}_i`

We want to determine the pose of the third camera:

- the global rotation :math:`\mathbf{R}`
- the global translation :math:`\mathbf{t}`

Projecting the 3D points to the image:

.. math::
   \mathbf{x}_i = \mathbf{K} [\mathbf{R} | \mathbf{t}] \mathbf{X}_i \\

   \tilde{\mathbf{x}}_i = \mathbf{R} \mathbf{X}_i + \mathbf{t} \\

   u_i = R

Thus we need :math:`n \geq 6` equations to fully retrieve the third camera pose and then solve
a least square problem.

We are then able to form a new bundle adjustment problem involving the three
cameras to refine again the 3D points and camera parameters (both external and
internal).

Proceeding incrementally like this, we can also retrieve the next camera poses.

REFERENCES
----------
- Lepetit et al.'s EPnP approach (IJCV 2008) which is better.
- Lambda-twist

Changing coordinate systems
===========================

Let us detail how we change coordinate from the world coordinate system to the
camera coordinates.

Firstly we denote the center of the world coordinate system by :math:`O_W` and
the center of the camera coordinate system :math:`O_C`.

Let :math:`M` denote a 3D point of the world, then


.. math::

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} + \overrightarrow{O_\textrm{C}M} \\

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} + x_C \mathbf{i}_C + y_C \mathbf{j}_C + z_C \mathbf{k}_C\\

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} +
     \left[ \begin{array}{c|c|c}
     \mathbf{i}_C & \mathbf{j}_C & \mathbf{k}_C
     \end{array} \right]

     \left[ \begin{array}{c} x_C \\ y_C \\ z_C \end{array} \right] \\

We recognise the global rigid body motion :math:`(\mathbf{R}, \mathbf{t})`
where:

.. math::
   \mathbf{t} = \overrightarrow{O_W O_C} \\

   \mathbf{R} = \left[ \mathbf{i}_C | \mathbf{j}_C | \mathbf{k}_C \right] \\

In other words, the rotation matrix :math:`\mathbf{R}` whose column vectors are
**the basis vectors of camera coordinate system expressed in the world coordinate
system**.


Euler Angles
============

The Euler angles :math:`(\psi, \theta, \phi)` are rotation angles about each
axis that describe a rotation matrix.

Let us first describe the rotation axes. Consider an airplane and the local
coordinate system attached to it. The usual axis convention in aerospace
engineering is as follows:

- The :math:`z`-axis is the longitudinal axis pointing to the airplane
  nose.
- The :math:`y`-axis is the transversal axis pointing to the right.
- The :math:`x`-axis is the vertical axis pointing downwards to the
  ground.

In terms of rotation,

- The :math:`z`-axis is the (current) yaw axis (NO head movement)
- The :math:`y`-axis is the (current) pich axis (YES head movement).
- The :math:`x`-axis is the (current) roll axis ("Indian" MAYBE head movement).

A 3D rotation can be decomposed into three elementary rotations in the following
order:

1. Yaw about the :math:`z`-axis by an angle :math:`\psi`,
2. Pitch about the current :math:`y'`-axis by an angle :math:`\theta`,
3. Roll about the current :math:`x''`-axis by an angle :math:`\phi`

Let us stress again the rotation order is very important.

The rotation matrix expressed with respect to the initial local coordinates is
calculated as:

.. math::

   \mathbf{R} (\psi, \theta, \phi) = \mathbf{R}_z (\psi)
                                     \mathbf{R}_y (\theta)
                                     \mathbf{R}_x (\phi)

Proof
-----

The composition of rotations is

.. math::

   \mathbf{\mathbf{R}} (\psi, \theta, \phi) = \mathbf{R}_{x''} (\phi)
                                              \mathbf{R}_{y'} (\theta)
                                              \mathbf{R}_{z} (\psi)

In the sequel, we will alleviate the notation by omitting the angles.

To obtain :math:`\mathbf{R}_{y'}`, we need to understand that the pitch rotation
is done about the current axis :math:`\mathbf{y}' = \mathbf{R}_z \mathbf{y}`.

First notice that the pitch rotation matrix expressed in the current coordinate
system has a very simple form: :math:`\mathbf{R}_y`. But we want the rotation
matrix expressed in the original coordinates. So how do we get it?

The key point to understand is that to obtain the coordinates back in the
original coordinate system, we need to multiply the coordinates in the current
coordinate system with the inverse rotation :math:`\mathbf{R}_y^T`.

Indeed if we project the coordinates of the original axes to the current axes
corresponds to the column vectors of the inverse rotation
:math:`\mathbf{R}_z^T`.

So consider a 3D point :math:`u`. Its coordinates w.r.t. the original axes is
:math:`\mathbf{u}_0 = u_{i}^0 \mathbf{e}^i` (Einstein notation), where
:math:`\mathbf{e}^i`, are the column vectors of the identity matrix
:math:`\mathbf{I}_3`.

By rotating the point :math:`u` yields, we create a new point :math:`v` where:

- In the original coordinate system, its coordinates are
  :math:`\mathbf{v}_0 = u_i^0 \mathbf{R}_z^i = \mathbf{R}_z \mathbf{u}_0`,
- In the current coordinate system, its coordinates are :math:`\mathbf{v}_1 = u_i^0 \mathbf{e}^i`.

The original axes has also moved to the current axes by the same rotation
:math:`\mathbf{R}_z`, and as highlighted by the Einstein notation, we recognize
that the coordinates of :math:`v` in the current coordinate system are also:

.. math::

   \mathbf{v}_1 = \mathbf{u}_0 = \mathbf{R}_z^T \mathbf{v}_0 \\

If we rotate the point :math:`v` by :math:`\mathbf{R}_y'`, we create a third
point :math:`w` where:

- In the current axes, its coordinates are :math:`\mathbf{w}_1 = \mathbf{R}_y \mathbf{v}_1`
- In the current axes, its coordinates are :math:`\mathbf{w}_1 = \mathbf{R}_y \mathbf{R}_z^T \mathbf{v}_0`

Denoting its coordinates in the original axes by :math:`\mathbf{w}_0`

  .. math::
    \mathbf{w}_1 = \mathbf{R}_z^T \mathbf{w}_0 \\
    \mathbf{w}_0 = \mathbf{R}_z \mathbf{w}_1 \\
    \mathbf{w}_0 = \mathbf{R}_z \mathbf{R}_y \mathbf{R}_z^T \mathbf{v}_0

We have just calculated the pitch rotation:

.. math::

   \mathbf{R}_{y'} = \mathbf{R}_{z}
                     \mathbf{R}_{y}
                     \mathbf{R}_{z}^T

Likewise the rotation :math:`\mathbf{R}_{x''}` is obtained as:

.. math::

   \mathbf{R}_{x''} = \mathbf{R} \mathbf{R}_{y} \mathbf{R}^T

where

.. math::

   \mathbf{R} = \mathbf{R}_z \mathbf{R}_{y}

By multiplying the three rotations, the inverse rotations will disappear and we get
the formula shown above.

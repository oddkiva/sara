Changing coordinate systems
===========================

Let us detail how we change coordinate from the world coordinate system
:math:`(O_W, \mathbf{i}_W, \mathbf{j}_W,\mathbf{k}_W)` to the
camera coordinates :math:`(O_C, \mathbf{i}_C, \mathbf{j}_C,\mathbf{k}_C)`.

Let :math:`M` denote a 3D point. Suppose that we know its coordinates in the
camera coordinate system :math:`(x_C, y_C, z_C)`. We can retrieve its
coordinates in the world coordinate system :math:`(x_W, y_W, z_W)` as follows

.. math::

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} + \overrightarrow{O_\textrm{C}M} \\

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} + x_C \mathbf{i}_C + y_C \mathbf{j}_C + z_C \mathbf{k}_C\\

   \overrightarrow{O_W M} = \overrightarrow{O_W O_C} +
     \left[ \begin{array}{c|c|c}
     \mathbf{i}_C & \mathbf{j}_C & \mathbf{k}_C
     \end{array} \right]

     \left[ \begin{array}{c} x_C \\ y_C \\ z_C \end{array} \right] \\

   \left[ \begin{array}{c} x_W \\ y_W \\ z_W \end{array} \right] =
   \overrightarrow{O_W O_C} +
     \left[ \begin{array}{c|c|c}
     \mathbf{i}_C & \mathbf{j}_C & \mathbf{k}_C
     \end{array} \right]

     \left[ \begin{array}{c} x_C \\ y_C \\ z_C \end{array} \right] \\

We recognize the global rigid body motion :math:`(\mathbf{R}, \mathbf{t})`
where:

.. math::
   \mathbf{t} = \overrightarrow{O_W O_C} \\

   \mathbf{R} = \left[ \mathbf{i}_C | \mathbf{j}_C | \mathbf{k}_C \right] \\

In other words, the rotation matrix :math:`\mathbf{R}` is the matrix whose column vectors are
**the coordinates of the basis vectors of camera coordinate system in the world coordinate
system**.


Euler Angles
============

The Euler angles :math:`(\psi, \theta, \phi)` are the angles about each
axis of the local coordinate system that are used to parameter any rotation
matrix.

Let us first describe the rotation axes. Consider an airplane and its local
coordinate system. The usual axis convention used in aerospace engineering is as
follows:

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

Let us stress that again the order to which we apply each elementary rotation is
very important.

The rotation matrix expressed with respect to the initial local coordinates is
calculated as:

.. math::

   \mathbf{R} (\psi, \theta, \phi) = \mathbf{R}_z (\psi)
                                     \mathbf{R}_y (\theta)
                                     \mathbf{R}_x (\phi)

Proof
-----

The composite rotation is

.. math::

   \mathbf{\mathbf{R}} (\psi, \theta, \phi) = \mathbf{R}_{x''} (\phi)
                                              \mathbf{R}_{y'} (\theta)
                                              \mathbf{R}_{z} (\psi)

In the sequel, we will alleviate the notation by omitting the angles.

To obtain :math:`\mathbf{R}_{y'}`, we need to understand that the pitch rotation
is done about the current axis :math:`\mathbf{y}' = \mathbf{R}_z \mathbf{y}`.

- Denote by :math:`(x, y, z)` the basis vectors of the original local coordinate
  system.
- Denote by :math:`(x', y', z')` the basis vectors of the current local
  coordinate system.

We can see that the intrinsic pitch rotation matrix expressed in the current
local coordinate system :math:`(x', y', z')` has the convenient form:
:math:`\mathbf{R}_y`. But we want the rotation matrix :math:`\mathbf{R}_{y'}` to
be expressed in the coordinate system :math:`(x, y, z)`. So how do we get it?

As we can see above, the key point to understand is that to go from the current
coordinates :math:`(x', y', z')` to the original coordinate system :math:`(x, y,
z)`, we need to multiply the coordinates in the current coordinate system with
the rotation :math:`\mathbf{R}_z`, which accounts for the missing angle offsets.

Let us detail this point to convince ourselves.

- In the current local coordinate system :math:`(x', y', z')`, the coordinates
  of the basis vectors :math:`(x', y', z')` are simply the column vectors
  :math:`\mathbf{e}^i` of the identity matrix :math:`\mathbf{I}_3`.
- In the original local coordinate system :math:`(x, y, z)`, the coordinates of
  the same basis vectors :math:`(x', y', z')` are precisely the column vectors
  :math:`\mathbf{R}_z^i` of the rotation matrix :math:`\mathbf{R}_z`.

Now consider any point :math:`u` of the airplane, if its
coordinates are :math:`\mathbf{u}'` in the current coordinate system :math:`(x',
y', z')`:

.. math::

   \mathbf{u}' = u_i' \mathbf{e}^i \ \text{(using Einstein's notation)}

Then its coordinates in the the original local coordinate system :math:`(x, y,
z)` are

.. math::

   \mathbf{u} = u_i' \mathbf{R}_z^i \\

And thus we recognize a matrix-vector multiplication

.. math::

   \mathbf{u} = \mathbf{R}_z \mathbf{u}'

If we rotate the point :math:`u` by :math:`\mathbf{R}_{y'}`, we create a second
point :math:`v` where:

- In the current coordinate system :math:`(x', y', z')`, its coordinates are
  simply

  .. math::

     \mathbf{v}' = \mathbf{R}_y \mathbf{u}'

- In the original coordinate system :math:`(x, y, z)`, its coordinates are
  :math:`\mathbf{v}`, thus by injecting the inverse rotation on both sides of
  the equality

  .. math::
    (\mathbf{R}_z^T \mathbf{v}) = \mathbf{R}_y\ (\mathbf{R}_z^T \mathbf{u})  \\
    \mathbf{v} = \mathbf{R}_z \mathbf{R}_y \mathbf{R}_z^T\ \mathbf{u}

We have just calculated the pitch rotation in the original coordinate system:

.. math::

   \mathbf{R}_{y'} = \mathbf{R}_{z}
                     \mathbf{R}_{y}
                     \mathbf{R}_{z}^T

Likewise the rotation :math:`\mathbf{R}_{x''}` is obtained as:

.. math::

   \mathbf{R}_{x''} = \mathbf{R} \mathbf{R}_{x} \mathbf{R}^T

where

.. math::

   \mathbf{R} = \mathbf{R}_z \mathbf{R}_{y}

And thus

.. math::

   \mathbf{R}_{x''} = \mathbf{R}_z \mathbf{R}_y \mathbf{R}_{x} \mathbf{R}_y^T \mathbf{R}_z^T


By multiplying the three rotations, the inverse rotations will disappear and we get
the formula as exposed in the Wikipedia page about Euler angles.

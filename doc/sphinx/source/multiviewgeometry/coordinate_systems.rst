Changing Coordinate Systems
===========================

Let us detail how we go from the world coordinates :math:`(O_W, \mathbf{i}_W,
\mathbf{j}_W,\mathbf{k}_W)` to the camera coordinates :math:`(O_C, \mathbf{i}_C,
\mathbf{j}_C,\mathbf{k}_C)`.

Denote a 3D point by :math:`M`. Suppose that we know its coordinates in the
camera coordinate system :math:`(O_C, x_C, y_C, z_C)`. We can retrieve its
coordinates in the world coordinate system :math:`(O_W, x_W, y_W, z_W)` as
follows

.. important::

   .. math::

     \begin{aligned}

     \overrightarrow{O_W M} &= \overrightarrow{O_W O_C} +
                               \overrightarrow{O_\textrm{C}M} \\

     \overrightarrow{O_W M} &= \overrightarrow{O_W O_C} +
                               x_C \mathbf{i}_C + y_C \mathbf{j}_C + z_C \mathbf{k}_C\\

     \overrightarrow{O_W M} &= \overrightarrow{O_W O_C} +
     \left[ \begin{array}{c|c|c}
     \mathbf{i}_C & \mathbf{j}_C & \mathbf{k}_C
     \end{array} \right]

     \left[ \begin{array}{c} x_C \\ y_C \\ z_C \end{array} \right] \\

     \left[ \begin{array}{c} x_W \\ y_W \\ z_W \end{array} \right] &=
       \underbrace{\overrightarrow{O_W O_C}}_{\mathbf{t}} +

       \underbrace{
         \left[ \begin{array}{c|c|c}
         \mathbf{i}_C & \mathbf{j}_C & \mathbf{k}_C
         \end{array} \right]
       }_{\mathbf{R}}

       \left[ \begin{array}{c} x_C \\ y_C \\ z_C \end{array} \right] \\
     \end{aligned}

We recognize the global rigid body motion :math:`(\mathbf{R}, \mathbf{t})`
where:

- The translation vector :math:`\mathbf{t}` is the camera origin coordinates expressed
  in the world coordinate system.
- The rotation matrix :math:`\mathbf{R}` are the coordinates of the camera basis
  vectors expressed in the world basis vectors.


Euler Angles
============

In engineering, we like to decompose any 3D rotation into Euler angles as they
are quite intuitive to understand. The usual convention that is followed is
called the *Tait-Bryan* convention, which we describe as follows.

Specifically the Euler angles :math:`(\psi, \theta, \phi)` are the three
rotational quantities about each *intrinsic* axis which altogether describes the
3D rotation.

Let us first describe the rotation axes by visualizing an airplane. Consider its
local coordinate system that is fixed to this airplane. The usual axis
convention used in aerospace engineering is as follows:

- The :math:`z`-axis is the longitudinal axis pointing to the airplane
  nose.
- The :math:`y`-axis is the transversal axis pointing to the right.
- The :math:`x`-axis is the vertical axis pointing downwards to the
  ground.

In terms of rotation,

- The :math:`z`-axis is the (current) yaw axis (NO head movement)
- The :math:`y`-axis is the (current) pich axis (YES head movement).
- The :math:`x`-axis is the (current) roll axis ("Indian" MAYBE head movement).

A 3D rotation can be decomposed into three elementary rotations that must
composed in the following order:

1. Yaw about the :math:`z`-axis of the airplane by an angle :math:`\psi`,
2. Pitch about the **current** :math:`y'`-axis of the airplane by an angle :math:`\theta`,
3. Roll about the **current** :math:`x''`-axis of the airplane by an angle :math:`\phi`

Once again let us stress the order to which we apply each elementary rotation is
very important.

The rotation matrix expressed with respect to the *extrinsic* axes (i.e., the axes
of the original local coordinates before rotating the plane) is calculated as:

.. important::

   .. math::

      \mathbf{R} (\psi, \theta, \phi) = \mathbf{R}_z (\psi)
                                        \mathbf{R}_y (\theta)
                                        \mathbf{R}_x (\phi)

Formula of Elementary Rotations
-------------------------------

Every now and then we may forget the formula of each elementary rotation and in
particular we don't remember their signs and orientations.

We can retrieve each coefficient of the matrix by reasoning about a small
positive quantity. A small angle will imply a small change, and then a cosine
close to 1 and a sine close to 0. By visualization, we can determine the sign of
each coefficient.

- When the airplane is yawing positively, it is steering to the right as the
  :math:`x`-axis is rotating towards the :math:`y`-axis:

  .. note::

     .. math::

       \mathbf{R}_z(\psi) = \left[ \begin{array}{ccc}
         \cos\psi & -\sin\psi & 0 \\
         \sin\psi &  \cos\psi & 0 \\
                0 &         0 & 1 \\
       \end{array} \right]

- When the airplane is pitching positively, its nose is looking up as the
  :math:`z`-axis is rotating towards the :math:`x`-axis.

  .. note::

     .. math::

       \mathbf{R}_y(\theta) = \left[ \begin{array}{ccc}
         \cos\theta & 0 & \sin \theta \\
                  0 & 1 &           0 \\
        -\sin\theta & 0 & \cos \theta \\
       \end{array} \right]

- When the airplane is rolling positively, it right wing is tilting down as the
  :math:`y`-axis is rotating towards the :math:`z`-axis.

  .. note::

     .. math::

       \mathbf{R}_x (\phi) = \left[ \begin{array}{ccc}
         1 &        0 &         0 \\
         0 & \cos\phi & -\sin\phi \\
         0 & \sin\phi &  \cos\phi \\
       \end{array} \right]


This nice visualisation tool can be useful to check our understanding:
http://danceswithcode.net/engineeringnotes/rotations_in_3d/demo3D/rotations_in_3d_tool.html

Proof
-----

It is useful to provide a proof that justifies the Euler decomposition we have
exhibited above.

In terms of matrix multiplication, the composite rotation is

.. math::

   \mathbf{\mathbf{R}} (\psi, \theta, \phi) = \mathbf{R}_{x''} (\phi)
                                              \mathbf{R}_{y'} (\theta)
                                              \mathbf{R}_{z} (\psi)

In the sequel, we will alleviate the notation by omitting the angles.

To obtain :math:`\mathbf{R}_{y'}`, we need to understand that the pitch rotation
is done about the current axis :math:`\mathbf{y}' = \mathbf{R}_z \mathbf{y}`.
And the vector :math:`\mathbf{y}'` is the coordinates of the current airplane :math:`y'`-axis
w.r.t. the original axes.

Now we denote

- the basis vectors of the original local coordinate system by
  :math:`(\mathbf{i}, \mathbf{j}, \mathbf{k})` and
- the basis vectors of the current local coordinate system by
  :math:`(\mathbf{i}', \mathbf{j}', \mathbf{k}')`.

In the sequel, we will alleviate the notation by omitting the origin :math:`O`
of the coordinate systems because there is no translation.

We can see that the intrinsic pitch rotation matrix expressed in the *current*
local coordinate system :math:`(x', y', z')` has the convenient form:

.. math::

  \mathbf{R}_y = \left[ \begin{array}{ccc}
    \cos\theta & 0 & \sin \theta \\
             0 & 1 &           0 \\
   -\sin\theta & 0 & \cos \theta \\
  \end{array} \right]

But we want the rotation matrix :math:`\mathbf{R}_{y'}` to be expressed in the
*original* coordinate system :math:`(x, y, z)`.  So how do we get it?

As we can see above, the key point to understand is that to go from the current
coordinates :math:`\mathbf{u}'` to the original coordinates :math:`\mathbf{u}`,
we need to multiply the current coordinates :math:`\mathbf{u}'` with the
rotation :math:`\mathbf{R}_z`, which "adds" the necessary angles offsets.

Let us detail this point to convince ourselves.

- In the current local coordinate system :math:`(x', y', z')`, the coordinates
  of the basis vectors :math:`(\mathbf{i}', \mathbf{j}', \mathbf{k}')` are
  simply the column vectors :math:`\mathbf{e}^i` of the identity matrix
  :math:`\mathbf{I}_3`.
- In the original local coordinate system :math:`(x, y, z)`, the coordinates of
  the same basis vectors :math:`(\mathbf{i}', \mathbf{j}', \mathbf{k}')` are
  precisely the column vectors :math:`\mathbf{R}_z^i` of the rotation matrix
  :math:`\mathbf{R}_z`.

Now consider any point :math:`u` of the airplane, if its
coordinates are :math:`\mathbf{u}'` in the current coordinate system :math:`(x',
y', z')`, then by linear combination

.. math::

   \mathbf{u}' = u_i' \mathbf{e}^i \ \text{(using Einstein's notation)}

Then its coordinates in the the original local coordinate system :math:`(x, y,
z)` are

.. math::

   \mathbf{u} = u_i' \mathbf{R}_z^i \\

And thus we recognize the matrix-vector multiplication

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

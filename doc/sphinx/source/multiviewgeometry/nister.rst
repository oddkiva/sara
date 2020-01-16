.. _chap-nister:


Nister's Method for the Relative Pose Estimation
================================================

This method solves the essential matrix from 5 point correspondences.

Underdetermined System of Linear Equations
------------------------------------------

Let :math:`(\mathbf{p}_i, \mathbf{q}_i)`, :math:`0 \leq i < 5` be :math:`5` point
correspondences, with :math:`\mathbf{p}_i` and :math:`\mathbf{q}_i` being points
on the left image and the right image respectively.

They satisfy the epipolar constraints:

.. math::

   \mathbf{q}_i^T \mathbf{E} \mathbf{p}_i = 0

Denoting the vectorized matrix :math:`\mathbf{E}` as
:math:`\mathbf{e} = \text{vec}(\mathbf{E}) = [\mathbf{E}[0, :], \mathbf{E}[1, :], \mathbf{E}[2, :]`

Expanding the system of five equations, we have:

.. math::
   \mathbf{Q} \mathbf{e} = 0

with

.. math::
   \mathbf{Q}[i, :] =  \text{vec}(\mathbf{p}_{i} \mathbf{q}_i^T)

:math:`\mathbf{e}` lives in the nullspace of :math:`\mathbf{A}` which is of
dimension :math:`4`.


Nullspace Extraction
--------------------

Using decompose the nullspace of :math:`\mathbf{A}` with basis vectors
:math:`(\mathbf{x}, \mathbf{y}, \mathbf{z}, \mathbf{w})` using singular value
decomposition, the vectorized essential matrix :math:`\mathbf{e}` is a linear
combination of these :math:`4` basis vectors:

.. math::
   \text{Null}(\mathbf{Q}) =
   \text{span}(\mathbf{x}, \mathbf{y}, \mathbf{z}, \mathbf{w}) \\

   \mathbf{e} = x \mathbf{x} + y \mathbf{y} + z \mathbf{z} + \mathbf{w}

By construction, the essential matrix :math:`\mathbf{E}` has rank 2, so its determinant
must be zero.

The essential matrix satisfies the following constraints:

.. math::
   \text{det}(\mathbf{E}) = 0 \\

   \mathbf{E} \mathbf{E}^T \mathbf{E} -
   \frac{1}{2} \text{trace}(\mathbf{E}\mathbf{E}^T) \mathbf{E} = 0

Plugging the linear combination in these system equations, we obtain a system of
:math:`10` equations in the monomial :math:`x^\alpha y^\beta z^\gamma`.


Gauss-Jordan Elimination
------------------------


Real Root Extraction
--------------------

Instead of using Sturm sequences, we extract the roots of the polynomials using
Jenkins-Traub algorithm.


Under construction...
---------------------

.. math::

  \text{Rendered with \KaTeX} \\[18pt]

  \gdef \f #1 {f(#1)}

  \f{x} = \int_{-\infty}^\infty
    \hat \f\xi\, e^{2 \pi i \xi x}
    \,d\xi

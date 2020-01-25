.. _chap-nister:


Nister's Method for the Relative Pose Estimation
================================================

This method solves the essential matrix from 5 point correspondences.

Underdetermined System of Linear Equations
------------------------------------------

Let :math:`(\mathbf{p}_i, \mathbf{q}_i)`, :math:`0 \leq i < 5` be :math:`5` point
correspondences, with :math:`\mathbf{p}_i` and :math:`\mathbf{q}_i` being points
on the left image and the right image respectively (in homogeneous coordinates).

They satisfy the epipolar constraints:

.. math::

   \bq_i^T \bE \bp_i = 0

Let :math:`\be` denote the essential matrix in its row-major vectorized
form, *i.e.*,

.. math::
   \begin{aligned}
   \be &= \text{vec}(\bE) \\

   \be &= \left[ \bE[0, :], \bE[1, :], \bE[2,:] \right]
   \; \text{(in numpy notation)}
   \end{aligned}


Expanding the system of five equations, we have equivalently:

.. math::
   \bQ \be = 0

with each row of :math:`\bQ` defined as

.. math::
   \bQ[i, :] =  \text{vec}(\mathbf{p}_{i} \mathbf{q}_i^T)

:math:`\be` lives in the nullspace of :math:`\mathbf{Q}` which is of
dimension :math:`4`. And the nullspace depends on the :math:`5` point
correspondences.

Nullspace Decomposition
-----------------------

We can decompose the nullspace :math:`\text{Null}(\mathbf{Q})` with basis vectors
:math:`(\mathbf{x}, \mathbf{y}, \mathbf{z}, \mathbf{w})` using singular value
decomposition, the vectorized essential matrix :math:`\be` is a linear
combination of these :math:`4` basis vectors:

.. math::
   \text{Null}(\bQ) = \text{span}(\x, \y, \z, \w) \\
   \be = x \x + y \y + z \z + \w

The goal is to determine the value of the real scalars :math:`x, y, z`.

Essential Matrix Constraints
----------------------------
As mentioned in Nister's paper:

1. the essential matrix is also characterized equivalently by the matrix
   equation:

   .. math::
      \begin{bmatrix}
      \langle e_{00} \rangle & \langle e_{01} \rangle & \langle e_{02} \rangle \\
      \langle e_{10} \rangle & \langle e_{11} \rangle & \langle e_{12} \rangle \\
      \langle e_{20} \rangle & \langle e_{21} \rangle & \langle e_{22} \rangle \\
      \end{bmatrix}
      \iff
      \bE \bE^T \bE - \frac{1}{2} \text{trace}(\bE\bE^T) \bE = 0

   This gives :math:`3 \times 3 = 9` polynomial equations :math:`\langle e_{ij}
   \rangle` in the monomials :math:`x^\alpha y^\beta z^\gamma` where
   :math:`\alpha + \beta + \gamma \leq 3`.

2. the essential matrix :math:`\bE` has rank 2, so its determinant must be
   zero:

   .. math::
      \langle d \rangle \iff \text{det}(\bE) = 0

   This gives another polynomial equation :math:`\langle d \rangle` in the
   monomials :math:`x^\alpha y^\beta z^\gamma`.

So we have in total :math:`10` polynomial equations in the monomials
:math:`x^\alpha y^\beta z^\gamma`. There are 20 monomials in total.

Nister enumerates these monomials in the following order:

.. math::

   \scriptsize

   \begin{array}{|c|cccccccccc|ccc|ccc|cccc|}
   \hline
     &
   x^3 & y^3 & x^2 y & x y^2 & x^2 z & x^2 & y^2 z & y^2 & xyz & xy &
   x & x z & x z^2 & y & y z & y z^2 & 1 & z & z^2 & z^3 \\

   \hline

   \langle e_{00} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{01} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{02} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{10} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{11} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{12} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{20} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{21} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \langle e_{22} \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \hline

   \langle d \rangle &
   . & . & . & . & . & . & . & . & . & . &
   . & . & . & . & . & . & . & . & . & . \\

   \hline
   \end{array}

In his paper, Nister does not use the Groebner basis to solve the system of
:math:`10` polynomial equations. Rather he uses elementary linear algebra
operations, which is very neat. We detail how he does it.


Gauss-Jordan Elimination
------------------------

Using the Gauss-Jordan elimination we can reduce the system of polynomial
equation so that the left block of the matrix above is zero everywhere except
one on the diagonal.

We will realize that it is actually sufficient to apply the Gauss-Jordan
elimination. Specifically,

1. perform the full sweep downward so that lower diagonal part
   is fully zero

   .. math::
      :label: eq-sweep-downward

      \scriptsize
      \begin{array}{|c|cccccccccc|ccc|ccc|cccc|}
      \hline

      &
      x^3 & y^3 & x^2 y & x y^2 & x^2 z & x^2 & y^2 z & y^2 & xyz & xy &
      x & x z & x z^2 & y & y z & y z^2 & 1 & z & z^2 & z^3 \\

      \hline

      \langle e_{00} \rangle &
      1 & . & . & . & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{01} \rangle &
      0 & 1 & . & . & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{02} \rangle &
      0 & 0 & 1 & . & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{10} \rangle &
      0 & 0 & 0 & 1 & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{11} \rangle &
      0 & 0 & 0 & 0 & 1 & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{12} \rangle &
      0 & 0 & 0 & 0 & 0 & 1 & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{20} \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 1 & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{21} \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{22} \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & . &
      . & . & . & . & . & . & . & . & . & . \\

      \hline

      \langle d \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 &
      . & . & . & . & . & . & . & . & . & . \\

      \hline
      \end{array}

   **TODO** write pseudo-code or sketch C++ code.

2. then in the sweep upward, stop halfway until the system of polynomial
   equations looks like the system of equations :eq:`eq-gaussjordan` below:

   .. math::
      :label: eq-gaussjordan

      \scriptsize
      \begin{array}{|c|cccccccccc|ccc|ccc|cccc|}
      \hline

      &
      x^3 & y^3 & x^2 y & x y^2 & x^2 z & x^2 & y^2 z & y^2 & xyz & xy &
      x & x z & x z^2 & y & y z & y z^2 & 1 & z & z^2 & z^3 \\

      \hline

      \langle e_{00} \rangle &
      1 & . & . & . & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{01} \rangle &
      0 & 1 & . & . & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{02} \rangle &
      0 & 0 & 1 & . & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{10} \rangle &
      0 & 0 & 0 & 1 & . & . & . & . & . & . &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{11} \rangle &
      0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{12} \rangle &
      0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{20} \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{21} \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 &
      . & . & . & . & . & . & . & . & . & . \\

      \langle e_{22} \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 &
      . & . & . & . & . & . & . & . & . & . \\

      \hline

      \langle d \rangle &
      0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 &
      . & . & . & . & . & . & . & . & . & . \\

      \hline
      \end{array}

   **TODO** write pseudo-code or sketch C++ code.


Let's look again at the last :math:`6` equations :eq:`eq-gaussjordan`. We can
again reduce it by multiplying by :math:`z` and subtracting as follows:

.. math::
   :label: eq-klm

   \begin{aligned}
   \langle k \rangle &= \langle e_{20} \rangle - z \langle e_{21} \rangle \\
   \langle l \rangle &= \langle e_{20} \rangle - z \langle e_{21} \rangle \\
   \langle m \rangle &= \langle e_{22} \rangle - z \langle d \rangle
   \end{aligned}

Now the system of equations :eq:`eq-klm` depends only the following groups of
monomials:

- :math:`\left\{ x, x z, x z^2, x z^3 \right\}`
- :math:`\left\{ y, y z, y z^2, x z^3 \right\}`
- :math:`\left\{ 1, z, z^2, z^3, z^4 \right\}`

The nice thing is that when we group coefficients using these three subgroups,
we see that:

- coefficients in the variable :math:`x` forms a polynomial in `z` of degree `3`,
- coefficients in the variable :math:`y` forms a polynomial in `z` of degree `3`,
- coefficients in the variable :math:`z` forms a polynomial in `z` of degree `4`.

In Nister's paper, the system of polynomial equations :eq:`eq-klm` is rewritten
as

.. math::
   :label: eq-B

   \begin{array}{|c|ccc|}
   \hline
   \mathbf{B} & x & y & 1 \\
   \hline
   \langle k \rangle & [3] & [3] & [4] \\
   \langle l \rangle & [3] & [3] & [4] \\
   \langle m \rangle & [3] & [3] & [4] \\
   \hline
   \end{array}

   \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \mathbf{0}_3

We can see that :math:`[x, y, 1]^T` is a vector in the nullspace of
:math:`\mathbf{B}` where the coefficients are polynomials in the variable
:math:`z` of degree :math:`3` or :math:`4`.

Equivalently, the determinant of :math:`\mathbf{B}` must be zero. This defines a
polynomial in the variable :math:`z` of degree :math:`10`.  Hence we can extract
the roots of this polynomial.

.. math::
   :label: eq-n

   \langle n \rangle \iff \text{det}(\mathbf{B}) = 0


Real Root Extraction
--------------------

Instead of using Sturm sequences, we extract the roots of the polynomials using
Jenkins-Traub algorithm.


Under construction...
~~~~~~~~~~~~~~~~~~~~~

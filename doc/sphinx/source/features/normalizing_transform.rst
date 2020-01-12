.. _chap:normtransform:

Normalizing Transform of a Feature
==================================

Let us remark the following proposition which relates the normalizing transform
:math:`\bT_x` to the feature shape :math:`\bSigma_x`.


.. important::

   Let :math:`L` be an invertible linear transformation in :math:`\mathbb{R}^2`
   whose matrix is denoted by :math:`\bL`.  For any point :math:`\x` in the
   zero-centered unit circle in :math:`\mathbb{R}^2`, its transformed point by
   :math:`L` is in the ellipse defined by 

   .. math::

      \left\{ \z \in \mathbb{R}^{2} | \z^T (\bL^{T})^{-1} \bL^{-1} \z = 1
      \right\}`


.. note::

   This note provides a proof of the proposition above.

   Fix a point :math:`\begin{bmatrix} \cos(t) \\ \sin(t) \end{bmatrix}` of
   the unit circle in :math:`\mathbb{R}^2`. We write its transformed point
   by :math:`L` as
   
   .. math::
   
      \begin{bmatrix} u \\ v \end{bmatrix} = \bL \begin{bmatrix} \cos(t) \\ \sin(t) \end{bmatrix}.
   
   Since :math:`\bL` is invertible
   
   .. math:: \bL^{-1} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \cos(t) \\ \sin(t) \end{bmatrix}
   
   The squared Euclidean norm of the equality yields
   
   .. math::
   
      \begin{bmatrix} u & v \end{bmatrix} (\bL^{-1})^T \bL^{-1} \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \cos(t) & \sin(t) \end{bmatrix} \begin{bmatrix} \cos(t) \\ \sin(t) \end{bmatrix} = 1
   
   We recognize the equation of an ellipse, which concludes the proof of
   proposition `[eq:lin_transform] <#eq:lin_transform>`__.


Geometric interpretation of the QR factorization
------------------------------------------------

Consider the shape matrix :math:`\bSigma_x`. Recall that
:math:`\bSigma_x` defines the elliptic shape :math:`\Shape_x`. We want
to retrieve the transformation :math:`L_x` that satisfies

.. math::
   :label: eq-sigma_l

   \bSigma_x = (\bL_x^{-1})^T \bL_x^{-1}.

Observe from the QR factorization :math:`\bL_x = \bQ \bR` that :math:`L_x` can
be decomposed uniquely in two specific transformations :math:`\bQ` and
:math:`\bR`. The upper triangular matrix :math:`\bR` encodes a transformation
that combines of shear and scaling transforms. The orthonormal matrix
:math:`\bQ` encode a rotation. This geometric interpretation is illustrated in
Figure [:ref:`fig-qrd`].

.. _fig-qrd:
.. figure:: ../figures/normalizingTransform.png
   :align: center

   Geometric interpretation of the QR factorization of linear transform
   matrix :math:`\bL_x`.

Unless :math:`L_x` involves no rotation, :math:`\bL_x` is an upper triangular
matrix. Then, because Equation :eq:`eq-sigma_l` is a Cholesky decomposition,
:math:`\bL_x` can be identified by unicity of the Cholesky decomposition.

In general, :math:`\bL_x` is not upper triangular. Orientations
:math:`\bo_x` of elliptic shape :math:`\bSigma_x` are provided from
feature detectors. In SIFT, :math:`\bo_x` corresponds to a dominant local
gradient orientation.

Thus, introducing :math:`\theta_x \eqdef \angle \left(
\begin{bmatrix}1\\0\end{bmatrix}, \bo_x \right)`,
we have

.. math::

   \bQ = \begin{bmatrix}
   \cos(\theta_x) & -\sin(\theta_x) \\
   \sin(\theta_x) &  \cos(\theta_x)
   \end{bmatrix}
   
and expanding Equation :eq:`eq-sigma_l` yields

.. math::

   \begin{aligned}
       \bSigma_x &= (\bL_x^{-1})^T \bL_x^{-1} \\
                 &= \bQ (\bR^{-1})^T \bR^{-1} \bQ^{T} \quad \text{since}\ \bQ^T = \bQ^{-1}\\
       \bQ^T \bSigma_x \bQ &=  (\bR^{-1})^T \bR^{-1}
   \end{aligned}

We recognize the Cholesky decomposition of matrix :math:`\bQ^T \bSigma_x \bQ`
which is the rotated ellipse as shown in Figure [:ref:`fig-qrd`], in which case
:math:`\bL_x` can be determined completely.

Finally, the affinity that maps the zero-centered unit circle to ellipse
:math:`\Shape_x` is of the form, in homogeneous coordinates

.. math:: \displaystyle \bT_x = \begin{bmatrix} \bL_x & \x \\ \mathbf{0}_2^T & 1 \end{bmatrix}.


Calculation of the Normalizing Transform
----------------------------------------

The algorithm below summarizes how to compute :math:`\bT_x`.

.. _alg-affT:
.. important::

   - Calculate the angle
   
     .. math::
   
        \theta_x :=
        \mathrm{atan2}\left(
        \left\langle \bo_x, \begin{bmatrix}0\\1\end{bmatrix}\right\rangle,
        \left\langle \bo_x, \begin{bmatrix}1\\0\end{bmatrix}\right\rangle
        \right)
           
   - Form the rotation matrix
     
     .. math:: 
   
        \bQ := \begin{bmatrix} 
        \cos(\theta_x) & -\sin(\theta_x) \\
        \sin(\theta_x) &  \cos(\theta_x)
        \end{bmatrix}
   
   - Decompose the ellipse matrix
     :math:`\bM := \mathrm{Cholesky}(\bQ^T \bSigma_x \bQ)`
   
   - :math:`\bM` is a lower triangular matrix such that
   
     - :math:`\bM \bM^T = \bQ^T \bSigma_x \bQ`
     - :math:`\bR := (\bM^T)^{-1}`
     - :math:`\bL := \bQ \bR`
     - :math:`\bT_x := \begin{bmatrix} \bL & \x \\ \mathbf{0}_2^T & 1 \end{bmatrix}`

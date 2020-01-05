Descriptors
===========

This section breaks down the SIFT descriptor that *Sara* implements.

In the sequel, let us denote an oriented scale-invariant local keypoint by
:math:`k = (x, y, \sigma, \theta)`.


SIFT
****

Interpretation
--------------

SIFT is a feature vector that encodes the photometric information of an image
patch into histograms of gradients.

The image patch is divided into a 2D grid of :math:`N \times N` square image
patches, from which we calculate histogram of gradients.  Each histogram bins
gradient orientations into :math:`O` principal orientations.  A single SIFT
descriptor about can be viewed as a 3D tensor :math:`\mathbf{h} \in
\mathbb{R}^{N \times N \times O}`.

The component :math:`\mathbf{h}[i, j, o]` quantifies the frequency of
orientation :math:`\frac{2 o \pi}{O}\ \text{rad}` in the :math:`(i, j)`-th
histogram, where :math:`{0 \leq i,j < N}` and :math:`0 \leq o < O`.

.. note::
   In the original implementation :cite:`Lowe:2004:ijcv` the parameters are set
   as :math:`N = 4` and :math:`O = 8`. And a SIFT descriptor is a feature vector
   :math:`\mathbf{h} \in \mathbb{R}^{128}`.

.. important::
   A peculiarity with SIFT is that image patches that form the grid *overlap*
   and that the orientation bins *overlap* as well because of the trilinear
   interpolation trick as we will see later.


Design Details
--------------

The design of SIFT is geared towards robustness w.r.t. *noisy* estimation of
keypoint position, scale and orientation, *lossy and noisy* image
acquisition, *illumination* changes, *etc*.

The SIFT can encode the photometric information of the region around the
keypoint :math:`k` in a similarity-invariant manner if we make use of the scale
:math:`\sigma` and the dominant gradient orientation :math:`\theta`.

By similarity invariance, we mean that

- if we take multiple photographs :math:`I_i` of the same salient region, *e.g.*,
  a corner or blob, and
- if we have a perfectly accurate feature detector and
  orientation estimator

then we would detect that salient region in each image as a keypoint :math:`k_i`
with perfect pixel localisation, scale and orientation.

Normalizing the image patch around this salient region with scale :math:`\sigma`
and orientation with :math:`\theta` would yield the same image. The SIFT
descriptors :math:`\mathbf{h}_i` simply compress the relevant image information
in each normalized patch around the keypoint :math:`k_i` and are thus expected
to be identical.

Grid of Overlapping Patches and Overlapping Histogram Bins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We dissect the anatomy of the SIFT descriptors further.  To compute a SIFT
descriptor, we consider an oriented square image patch :math:`\mathcal{P}`:

- centered in :math:`\mathbf{x} = (x, y)` and,
- oriented with an angle :math:`\theta` w.r.t. the image axes to enforce
  invariance to rotation,
- with a side length proportional to the scale :math:`\sigma` to enforce
  invariance to scale change.

.. note::
   We divide the square patch :math:`\mathcal{P}` into a grid of :math:`N \times
   N` **overlapping** square patches :math:`\mathcal{P}_{ij}` for :math:`{0 \leq
   i,j < N},` each of them having a *radius* :math:`s = \lambda_{\text{zoom}}
   \sigma` pixels.  This means that the whole square patch :math:`\mathcal{P}`
   has a side length equal to :math:`(N + 1) s`

   In the original implementation of :cite:`Lowe:2004:ijcv` the magnification factor
   :math:`\lambda_{\text{zoom}}` is set to :math:`3`.

We stress that the patches do **overlap**. We will see later that the trilinear
interpolation used in :cite:`Lowe:2004:ijcv` amounts to doing this. And this is
one key ingredient to compensate for the **noisy** estimation of keypoint
:math:`k`. This means that a pixel in the patch can belong up to :math:`4`
adjacent patches :math:`\mathcal{P}_{i'j'}` for :math:`i \leq i' \leq i + 1` and
:math:`j \leq j' \leq j + 1`.

We quote :cite:`Lowe:2004:ijcv` to explain the rationale behind this: *"It is
important to avoid all boundary affects in which the descriptor abruptly changes
as a sample shifts smoothly from being within one histogram to another or from
one orientationto another."*

Then we encode the photometric information of each patch
:math:`\mathcal{P}_{ij}` into a histogram :math:`\mathbf{h}_{ij} \in
\mathbb{R}^O` of *gradient orientations*, where the orientations are binned into
:math:`O` principal orientations. Again, because of the trilinear interpolation,

.. note::
   The histogram bins :math:`\mathbf{h}[i, j, o]` **overlap** as each of them
   covers the interval of orientations :math:`[\frac{2 \pi (o - 1)}{O}, \frac{2
   \pi (o + 1)}{O}]`.


Local Coordinate System and Normalizing Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The orientations are calculated and accumulated with respect to the **local
coordinate system** :math:`\mathcal{C}_k` associated to keypoint :math:`k`. This
local coordinate system is equivalently characterized by a similarity transform
:math:`\mathbf{T}` which we define as follows.

.. important::

   In matrix notation and using homogeneous coordinates, the normalizing
   transform :math:`\mathbf{T}` transforms *image coordinates* :math:`\mathbf{u}
   = (u, v)` to *normalized coordinates* :math:`\tilde{\mathbf{u}} = (\tilde{u},
   \tilde{v})` as follows

   .. math::

      \begin{bmatrix} \tilde{\mathbf{u}} \\ 1 \end{bmatrix}
      = \mathbf{T} \begin{bmatrix} \mathbf{u} \\ 1 \end{bmatrix}
      = \displaystyle
        \frac{1}{s}
        \left[
        \begin{array}{c|c}
        \mathbf{I}_2   & \mathbf{0}_2 \\[0.1em]
        \hline
        \\[-1em]
        \mathbf{0}_2^T & s
        \end{array}
        \right]

        \left[
        \begin{array}{c|c}
        \mathbf{R}_{\theta}^T & -\mathbf{R}_{\theta}^T \mathbf{x} \\[0.1em]
        \hline \\[-1em]
        \mathbf{0}_2^T & 1
        \end{array}
        \right]

        \begin{bmatrix} \mathbf{u} \\ 1 \end{bmatrix}


Geometry of Overlapping Patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this local coordinate system :math:`\mathcal{C}_k`:

- the keypoint center is at the origin :math:`(0, 0)`,
- each patch :math:`\mathcal{P}_{ij}` can be viewed as a patch with side length
  :math:`2`,
- each patch has:

  - its top-left corner at

    .. math::

       \mathbf{c}_{ij}^{\text{tl}} = [j, i] - \frac{N + 1}{2} [1, 1]

  - its bottom-right corner at

    .. math::

       \mathbf{c}_{ij}^{\text{br}} = [j, i] - \frac{N - 3}{2} [1, 1]

  - its center at

    .. math::

       \mathbf{c}_{ij} = [j, i] - \frac{N - 1}{2} [1, 1]

.. note::
   Clearly a patch :math:`\mathcal{P}_{ij}` is a closed ball centered in
   :math:`\mathbf{c}_{ij}` and with radius :math:`1` for the :math:`\ell_1`
   norm in this coordinate system.

   The patches overlap by construction because the centers are laid on a 2D grid
   with step size :math:`1`. The overlapping helps to make the SIFT descriptor
   more robust to the noisy estimation of the keypoint position, scale and
   orientation.

   The patch centers
   :math:`\mathbf{c}_{ij}` coincide with the histogram indices :math:`(i, j)` if
   we shift each coordinate with :math:`\frac{N - 1}{2}`.

   This observation will be useful in the SIFT implementation to determine which
   histogram bins needs to be accumulated.

Thus the centers are

.. math::
   \left[
   \begin{array}{c|c|c|c}
     (-1.5,-1.5) & (-0.5,-1.5) & (+0.5,-1.5) & (+1.5,-1.5) \\
     \hline
     (-1.5,-0.5) & (-0.5,-0.5) & (+0.5,-0.5) & (+1.5,-0.5) \\
     \hline
     (-1.5,+0.5) & (-0.5,+0.5) & (+0.5,+0.5) & (+1.5,+0.5) \\
     \hline
     (-1.5,+1.5) & (-0.5,+1.5) & (+0.5,+1.5) & (+1.5,+1.5) \\
   \end{array}
   \right]\\

Or in a *numpy-array*-like notation:

.. math::
   \left[
   \begin{array}{c|c|c|c}
     [0, 0] & [1, 0] & [2, 0] & [3, 0] \\
     \hline
     [0, 1] & [1, 1] & [2, 1] & [3, 1] \\
     \hline
     [0, 2] & [1, 2] & [2, 2] & [3, 2] \\
     \hline
     [0, 3] & [1, 3] & [2, 3] & [3, 3]
   \end{array}
   \right]
   - [1.5, 1.5]


SIFT Coordinate System
~~~~~~~~~~~~~~~~~~~~~~
Let us consider a pixel :math:`(u, v)` in the patch :math:`\mathcal{P}`:

.. math::
   \left\{
   \begin{aligned}
   & \displaystyle x - r \leq u \leq x + r \\
   & \displaystyle y - r \leq v \leq y + r \\
   \end{aligned}
   \right.

   \Longleftrightarrow

   \left\{
   \begin{aligned}
   & \displaystyle -\frac{N + 1}{2} \leq \tilde{u} \leq \frac{N + 1}{2} \\
   & \displaystyle -\frac{N + 1}{2} \leq \tilde{v} \leq \frac{N + 1}{2}
   \end{aligned}
   \right.

Introducing shifted coordinates which I choose to call these **SIFT
coordinates** :math:`\hat{\mathbf{u}} = (\hat{u}, \hat{v})`

.. math::
   \left\{
   \begin{aligned}
   \hat{u} &= \tilde{u} + \frac{N - 1}{2} \\
   \hat{v} &= \tilde{v} + \frac{N - 1}{2}
   \end{aligned}
   \right.

We see equivalently

.. math::
   \left\{
   \begin{aligned}
   -1 \leq \hat{u} \leq N \\
   -1 \leq \hat{v} \leq N
   \end{aligned}
   \right.

.. note::
   In matrix notation and using homogeneous coordinates, the normalizing
   transform :math:`\mathbf{T}_\text{SIFT}` transforms *image coordinates*
   :math:`\mathbf{u}` to *SIFT coordinates* :math:`\hat{\mathbf{u}}` as follows

   .. math::

      \begin{bmatrix} \hat{\mathbf{u}} \\ 1 \end{bmatrix}
      = \mathbf{T}_{\text{SIFT}} \begin{bmatrix} \mathbf{u} \\ 1 \end{bmatrix}
      = \displaystyle
        \underbrace
        {\left[
        \begin{array}{ccc}
        1 & 0 & \frac{N - 1}{2} \\[0.1em]
        0 & 1 & \frac{N - 1}{2} \\[0.1em]
        0 & 0 &              1  \\[0.1em]
        \end{array}
        \right]}_{\text{shift}}

        \mathbf{T}
        \begin{bmatrix} \mathbf{u} \\ 1 \end{bmatrix}

The floored coordinates satisfies:

.. math::

   \left\{
   \begin{aligned}
   -1 \leq \lfloor \hat{u} \rfloor \leq N \\
   -1 \leq \lfloor \hat{v} \rfloor \leq N
   \end{aligned}
   \right.

.. math::

   \left\{
   \begin{aligned}
   \lfloor \hat{u} \rfloor \leq \hat{u} < \lfloor \hat{u} \rfloor + 1 \\
   \lfloor \hat{v} \rfloor \leq \hat{v} < \lfloor \hat{v} \rfloor + 1
   \end{aligned}
   \right.

The pixel :math:`(u, v)` belongs up to :math:`4` patches:

- :math:`\mathcal{P}_{ \lfloor \hat{v} \rfloor    , \lfloor \hat{u} \rfloor     }`
- :math:`\mathcal{P}_{ \lfloor \hat{v} \rfloor    , \lfloor \hat{u} \rfloor  + 1}`
- :math:`\mathcal{P}_{ \lfloor \hat{v} \rfloor + 1, \lfloor \hat{u} \rfloor     }`
- :math:`\mathcal{P}_{ \lfloor \hat{v} \rfloor + 1, \lfloor \hat{u} \rfloor  + 1}`

We say "up to :math:`4`" because for example a gradient at the boundary
:math:`(-1,-1)` contributes only to :math:`P_{00}`.

Histogram of Gradients
~~~~~~~~~~~~~~~~~~~~~~

Consider a pixel :math:`\mathbf{u} \in \mathcal{P}_{ij}`. Its contribution in
histogram :math:`\mathbf{h}_{ij}` is

.. math::
   w(\mathbf{u}) =
   \underbrace{
   \exp \left( - \frac{\| \tilde{\mathbf{u}} \|^2}{2 (N/2)^2} \right)
   }_{\text{distance to center}}

   \underbrace{
   \| \nabla g_\sigma * I(\mathbf{u}) \|_2
   }_{\text{gradient magnitude}}

:cite:`Lowe:2004:ijcv` chooses to give more emphasis to gradients close to the
keypoint center :math:`\mathbf{x}` to compensate for the noisy estimation of
keypoint.

Using trilinear interpolation, its contribution to :math:`\mathbf{h}_{ij}`
becomes:

.. math::
   \displaystyle
   w_{\text{final}}(\mathbf{u}) = w(\mathbf{u})
                                  \left( 1 + i - \hat{v} \right)
                                  \left( 1 + j - \hat{u} \right)

The orientation of the gradient :math:`\nabla I_\sigma(\mathbf{u})` is
calculated as:

.. math::
   \phi = \text{atan2}(\nabla I_\sigma(\mathbf{u})) - \theta \\

The normalized orientation is:

.. math::
   \hat{\phi} = \phi \frac{O}{2 \pi}

It falls within two orientation bins:

- :math:`\mathbf{h}[i, j, o]`
- :math:`\mathbf{h}[i, j, o+1]`

where :math:`o = \lfloor \hat{\phi} \rfloor`.

The contribution will be distributed to the two bins as follows

.. math::
   \displaystyle
   \mathbf{h}[i, j, o] \leftarrow \mathbf{h}[i, j, o] +
   w_\text{final}(\mathbf{u}) \left(1 - (\hat{\phi} - o) \right) \\

   \displaystyle
   \mathbf{h}[i, j, o + 1] \leftarrow \mathbf{h}[i, j, o + 1] +
   w_\text{final}(\mathbf{u}) \left( \hat{\phi} - o \right)

.. note::
   The histogram bins have an explicit formula but it is not efficient to
   calculate it as is:

   .. math::
      \displaystyle
      \mathbf{h}[i, j, o] = \sum_{\mathbf{u} \in \mathcal{P}_{ij}}
      w(\mathbf{u})
      \left( 1 + i - \hat{u} \right)
      \left( 1 + j - \hat{v} \right)
      \left( 1 + o - \hat{\phi} \right)
      \mathbf{1}_{\left|\ \hat{\phi} - o \right| < 1}



Sketch of Implementation
------------------------

The last paragraph gives enough insights as for how to compute the SIFT
descriptor. It is easy to show that we need to scan all the pixels on a large
enough image patch, e.g., radius

.. math::

   r = \sqrt{2} \frac{N + 1}{2} \lambda_{\text{zoom}} \sigma

In the above formula, notice that:

- the factor :math:`\sqrt{2}`: because the square patches are oriented with an
  angle :math:`\theta \neq 0`, we need to make sure we are not missing any
  pixels in particular the corners of the patches; granted we consider pixels
  that are outside the patch domain and possibly there could be many of them
  that need to be discarded.
- if the orientation :math:`\theta` was zero, we could check that a radius
  :math:`r = \frac{N + 1}{2} \lambda_{\text{zoom}} \sigma` would have been
  sufficient.
- the factor :math:`\frac{(N + 1)}{2}`: this accounts for gradients for patches
  "at the border" of the image region. These gradients "at the border" may
  belong to only one histogram ("at the corners") or two histograms (at the
  edges).

The SIFT descriptor for keypoint :math:`k` is calculated as follows:

.. important::

   - For each pixel :math:`\mathbf{u} = (u, v) \in [x-r,x+r] \times [y-r, y+r]`:

     - calculate the gradient
       :math:`\nabla g_\sigma * I (\mathbf{u})`

     - express its orientation w.r.t. the local coordinate system
       :math:`\mathcal{C}_k`

     - calculate the contribution :math:`w(\mathbf{u})` of the gradient.

     - accumulate histograms using trilinear interpolation (*to be cont'd*)

.. note::
   In practice the gradients are precomputed only once in polar coordinates for
   efficiency at every scale of the Gaussian pyramid.

We can implement the computation of SIFT in *C++* as follows:

.. code-block:: cpp

    using descriptor_type = Eigen::Matrix<float, 128, 1>;

    auto compute_sift_descriptor(float x, float y, float sigma, float theta,
                                 const Image<Vector2f, 2>& grad_polar_coords)
        -> descriptor_type
    {
      static constexpr auto lambda_zoom = 3.f;

      // The SIFT descriptor.
      descriptor_type h = descriptor_type::Zero();

      // The radius of each overlapping patches.
      const auto s = lambda_zoom * sigma;

      // The radius of the total patch.
      const auto r = sqrt(2.f) * s * (N + 1) / 2.f;

      // Linear part of the normalization transform.
      const Eigen::Matrix2f T = Eigen::Rotation2D<float>{-theta}::toRotationMatrix() / s;

      // Loop to perform interpolation
      const auto rounded_r = static_cast<int>(std::round(r));
      const auto rounded_x = static_cast<int>(std::round(x));
      const auto rounded_y = static_cast<int>(std::round(y));
      for (auto v = -rounded_r; v <= rounded_r; ++v)
      {
        for (auto u = -rounded_r; u <= rounded_r; ++u)
        {
          // Retrieve the normalized coordinates.
          const Vector2f pos = T * Vector2f(u, v);

          // Boundary check.
          if (rounded_x + u < 0 || rounded_x + u >= grad_polar_coords.width() ||
              rounded_y + v < 0 || rounded_y + v >= grad_polar_coords.height())
            continue;

          // Gaussian weight contribution.
          const auto weight = exp(-pos.squaredNorm() / (2.f * pow(N / 2.f, 2)));

          // Read the precomputed gradient (in polar coordinates).
          const auto mag = grad_polar_coords(rounded_x + u, rounded_y + v)(0);
          auto ori = grad_polar_coords(rounded_x + u, rounded_y + v)(1) - theta;

          // Normalize the orientation.
          ori = ori < 0.f ? ori + 2.f * pi : ori;
          ori *= float(O) / (2.f * pi);

          // Shift the coordinates to retrieve the "SIFT" coordinates.
          pos.array() += N / 2.f - 0.5f;

          // Discard pixels that are not in the oriented patch.
          if (pos.minCoeff() <= -1.f || pos.maxCoeff() >= static_cast<float>(N))
            continue;

          // Accumulate the 4 gradient histograms using trilinear interpolation.
          trilinear_interpolation(h, pos, ori, weight, mag);
        }
      }

      return h;
    }


Trilinear Interpolation
~~~~~~~~~~~~~~~~~~~~~~~

We can implement the trilinear interpolation in C++ as follows:

.. code-block:: cpp

    void trilinear_interpolation(descriptor_type& h, const Vector2f& pos, float ori,
                                 float weight, float mag)
    {
      const auto xfrac = pos.x() - floor(pos.x());
      const auto yfrac = pos.y() - floor(pos.y());
      const auto orifrac = ori - floor(ori);
      const auto xi = static_cast<int>(pos.x());
      const auto yi = static_cast<int>(pos.y());
      const auto orii = static_cast<int>(ori);

      auto at = [](int y, int x, int o) { return y * N * O + x * O + o; };

      for (auto dy = 0; dy < 2; ++dy)
      {
        const auto y = yi + dy;
        if (y < 0 || y >= N)
          continue;

        const auto wy = (dy == 0) ? 1.f - yfrac : yfrac;
        for (auto dx = 0; dx < 2; ++dx)
        {
          const auto x = xi + dx;
          if (x < 0 || x >= N)
            continue;

          const auto wx = (dx == 0) ? 1.f - xfrac : xfrac;
          for (auto dori = 0; dori < 2; ++dori)
          {
            const auto o = (orii + dori) % O;
            const auto wo = (dori == 0) ? 1.f - orifrac : orifrac;
            // Trilinear interpolation:
            h[at(y, x, o)] += wy * wx * wo * weight * mag;
          }
        }
      }
    }

Robustness to illumination changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:cite:`Lowe:2004:ijcv` explains that:

- a brightness change consists in adding a constant factor to image intensities.
  And image gradients cancels this constant factor so SIFT is invariant to
  brightness change by construction.
- a contrast change in image amounts to multiplying image intensities by a
  constant factor. Normalizing the descriptor cancels the multiplication factor.
  So we must normalize the descriptor at the end.
- There are still other nonlinear illumination changes (camera saturation and
  surface reflective properties). :cite:`Lowe:2004:ijcv` have found
  experimentally that (1) clamping histogram bins to :math:`0.2` and then (2)
  renormalizing the descriptor again worked well to account for these on a
  specific dataset consisting of 3D objects photographed under different
  lighting conditions.

Using *Eigen*, we can express these in *C++*:

.. code-block:: cpp

   void normalize(descriptor_type& h)
   {
     // SIFT is by construction invariant to brightness change since it is based
     // on gradients.

     // Make the descriptor robust to contrast change.
     h.normalize();

     // Apply the following recipe for nonlinear illumination change.
     //
     // 1) Clamp the histogram bin values to 0.2
     h = h.cwiseMin(descriptor_type::Ones() * _max_bin_value);
     // 2) Renormalize again.
     h.normalize();
   }

Descriptors
===========


SIFT
****

Interpretation
--------------

SIFT is a feature vector that encodes the photometric information of an image
patch into histograms of gradients.

The image patch is divided into a 2D grid of :math:`N \times N` **overlapping**
square image patches, from which we calculate histogram of gradients.

Each histogram bins gradient orientations into :math:`O` principal orientations.
A single SIFT descriptor about can be viewed as a 3D tensor :math:`\mathbf{h}
\in \mathbb{R}^{N \times N \times O}`.

The component :math:`\mathbf{h}[i, j, o]` quantifies the frequency of
orientation :math:`\frac{2 o \pi}{O}\ \text{rad}` in the :math:`(i, j)`-th
histogram, where :math:`{0 \leq i,j < N}` and :math:`0 \leq o < O`.

In the original implementation [Lowe, IJCV 2004] the parameters are set as
:math:`N = 4` and :math:`O = 8`. And a SIFT descriptor is a feature vector
:math:`\mathbf{h} \in \mathbb{R}^{128}`.


Design and Implementation Details
---------------------------------

Histogram of Gradients
~~~~~~~~~~~~~~~~~~~~~~

Let us denote an oriented scale-invariant local keypoint by
:math:`k = (x, y, \sigma, \theta)`.

The SIFT can encode the photometric information of the region around the
keypoint :math:`k` in a similarity-invariant manner if we make use of the scale
:math:`\sigma` and the dominant gradient orientation :math:`\theta`.

To compute a SIFT descriptor, we consider a square image patch
:math:`\mathcal{P}`:

- centered in :math:`\mathbf{x} = (x, y)` and,
- with a side length proportional to the scale :math:`\sigma` to enforce
  invariance to scale change,
- oriented with an angle :math:`\theta` w.r.t. the image axes to enforce
  invariance to rotation.

We divide the square patch :math:`\mathcal{P}` into a grid of :math:`N \times N`
**overlapping** square patches :math:`\mathcal{P}_{ij}` for :math:`{0 \leq i,j <
N},` each of them having a radius equal to :math:`\lambda_{\text{zoom}} \sigma`
pixels:

We encode the photometric information of each patch :math:`\mathcal{P}_{ij}`
into a histogram :math:`\mathbf{h}_{ij} \in \mathbb{R}^O` of *gradient
orientations*, where the orientations are binned into :math:`O` principal
orientations. The orientations are calculated with respect to the **oriented**
*coordinate system*

.. math::

    \mathcal{C}_k =
      \left(\mathbf{x}, \mathbf{u}_\theta, \mathbf{v}_{\theta} \right)
to ensure invariance to rotation.


In the original implementation of [Lowe, IJCV 2004] the magnification factor
:math:`\lambda_{\text{zoom}}` is set to :math:`3`.


Keypoint coordinate system
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this oriented coordinate system :math:`\mathcal{C}_k`:

- the keypoint center is at the origin :math:`(0, 0)`,
- each patch :math:`P_{ij}` can be viewed as a patch with side length
  :math:`2 \lambda_{\text{zoom}} \sigma`,
- if the unit of measure is the patch size :math:`\lambda_{\text{zoom}} \sigma`,
  each patch has:

  - its top-left corner at

    .. math::

        [-\frac{N}{2} + i, -\frac{N}{2} + j]  - \frac{1}{2} [1, 1]


  - its bottom-right corner at

    .. math::

        [-\frac{N}{2} + i, -\frac{N}{2} + j] +  \frac{3}{2} [1, 1]


  - its center at

    .. math::

        C_{ij} = [-\frac{N}{2} + i, -\frac{N}{2} + j]  + \frac{1}{2}[1, 1]

Thus the centers are:

.. math::

    \left[
    \begin{array}{cccc}
      (-1.5,-1.5) & (-0.5,-1.5) & (0.5,-1.5) & (1.5,-1.5) \\
      (-1.5,-0.5) & (-0.5,-0.5) & (0.5,-0.5) & (1.5,-0.5) \\
      (-1.5, 0.5) & (-0.5, 0.5) & (0.5, 0.5) & (1.5, 0.5) \\
      (-1.5, 1.5) & (-0.5, 1.5) & (0.5, 1.5) & (1.5, 1.5) \\
    \end{array}
    \right]


The overlapping patches helps to make the SIFT descriptor more robust to
measurement uncertainty w.r.t. the estimation of the keypoint position, scale
and orientation.

To compute the SIFT descriptor we need to scan all the pixels
on a large enough image patch, e.g., radius

.. math::

    r = \sqrt{2} \frac{N + 1}{2}
        \lambda_{\text{zoom}} \sigma

.. code-block:: cpp

    const auto l = lambda_zoom * sigma;
    const auto r = sqrt(2.f) * l * (N + 1) / 2.f;


In the above formula, notice that:

- the factor :math:`\sqrt{2}`: because the square patches are oriented with an
  angle :math:`\theta \neq 0`, we need to make sure we are not missing any
  pixels in particular the corners of the patches;
- the factor :math:`\frac{(N + 1)}{2}`: this accounts for gradients for patches
  "at the border" of the image region.

The SIFT descriptor for keypoint :math:`k` is calculated as follows:

- For each pixel :math:`(x+u, y+v)`, with :math:`(u,v) \in [-r,r]^2`:

  - calculate the gradient
    :math:`\left. \nabla g_\sigma * I \right|_{(x + u, y + v)}`
  - express its gradient in the coordinate system :math:`\mathcal{C}_k` by
    applying the inverse transform
    :math:`T = \frac{1}{\lambda_{\text{zoom}} \sigma} R_\theta^T`.

.. code-block:: cpp

    // Histogram calculation of each patch (i, j).
      auto T = Matrix2f{};
      T << cos(theta), sin(theta),
          -sin(theta), cos(theta);
      T /= l;

      // Loop to perform interpolation
      const int rounded_r = int_round(r);
      const int rounded_x = int_round(x);
      const int rounded_y = int_round(y);
      for (auto v = -rounded_r; v <= rounded_r; ++v)
      {
        for (auto u = -rounded_r; u <= rounded_r; ++u)
        {
          // Compute the coordinates in the rescaled and oriented coordinate
          // frame bound to patch $P(k)$.
          auto pos = Vector2f{T * Vector2f(u, v)};
          // subpixel correction?
          /*pos.x() -= (x - rounded_x);
          pos.y() -= (y - rounded_y);*/

          if (rounded_x + u < 0 || rounded_x + u >= grad_polar_coords.width() ||
              rounded_y + v < 0 || rounded_y + v >= grad_polar_coords.height())
            continue;


Compute the Gaussian weight to give more emphasis to gradients close to the
center :math:`C_{ij}`.


.. code-block:: cpp

    // Cont'd...

          auto weight = exp(-pos.squaredNorm() / (2.f * pow(N / 2.f, 2)));
          auto mag = grad_polar_coords(rounded_x + u, rounded_y + v)(0);
          auto ori = grad_polar_coords(rounded_x + u, rounded_y + v)(1) - theta;
          ori = ori < 0.f ? ori + 2.f * pi : ori;
          ori *= float(O) / (2.f * pi);


The coordinate frame is centered in the patch center, thus:

.. math::

    (x, y) \in [-(N + 1) / 2, (N + 1) / 2]^2.

Translate the so that :math:`(x,y) \in [-1, N]^2`. Thus,
translate by :math:`[(N - 1) / 2, (N - 1)/ 2]`.


.. code-block:: cpp

    // Cont'd...

          pos.array() += N / 2.f - 0.5f;

          if (pos.minCoeff() <= -1.f || pos.maxCoeff() >= static_cast<float>(N))
            continue;
          // In the translated coordinate frame, note that for $N=4$ the centers
          // are now located at:
          //   (0,0) (1,0) (2,0) (3,0)
          //   (0,1) (1,1) (2,1) (3,1)
          //   (0,2) (1,1) (2,2) (3,2)
          //   (0,3) (1,1) (2,3) (3,3)
          //

          // Update the SIFT descriptor using trilinear interpolation.
          accumulate(h, pos, ori, weight, mag);
        }
      }

      h.normalize();

      h = (h * 512.f).cwiseMin(Matrix<float, Dim, 1>::Ones() * 255.f);
      return h;
    }


The trilinear interpolation is written as follows:

in a coordinate system transformed by a similarity, a gradient with orientation
:math:`\theta` and located at
:math:`(x,y) \in [-1,N]^2` contributes to the 4 histograms:

 - :math:`\mathbf{h}_{ \lfloor y \rfloor    , \lfloor x \rfloor     }`
 - :math:`\mathbf{h}_{ \lfloor y \rfloor    , \lfloor x \rfloor  + 1}`
 - :math:`\mathbf{h}_{ \lfloor y \rfloor + 1, \lfloor x \rfloor     }`
 - :math:`\mathbf{h}_{ \lfloor y \rfloor + 1, \lfloor x \rfloor  + 1}`

In each of these histograms, the following bins are accumulated:

 - :math:`\mathbf{h}_{o}`
 - :math:`\mathbf{h}_{o+1}`

where :math:`o = \lfloor \frac{\theta}{2 \pi} \times O \rfloor`

Note that a gradient at the boundary like :math:`(-1,-1)` contributes only
to :math:`P_{00}`.

.. code-block:: cpp

    void accumulate(descriptor_type& h, const Vector2f& pos, float ori,
                    float weight, float mag) const
    {
      const auto xfrac = pos.x() - floor(pos.x());
      const auto yfrac = pos.y() - floor(pos.y());
      const auto orifrac = ori - floor(ori);
      const auto xi = static_cast<int>(pos.x());
      const auto yi = static_cast<int>(pos.y());
      const auto orii = static_cast<int>(ori);

      for (auto dy = 0; dy < 2; ++dy)
      {
        auto y = yi + dy;
        if (y < 0 || y >= N)
          continue;

        auto wy = (dy == 0) ? 1.f - yfrac : yfrac;
        for (auto dx = 0; dx < 2; ++dx)
        {
          auto x = xi + dx;
          if (x < 0 || x >= N)
            continue;
          auto wx = (dx == 0) ? 1.f - xfrac : xfrac;
          for (auto dori = 0; dori < 2; ++dori)
          {
            auto o = (orii + dori) % O;
            auto wo = (dori == 0) ? 1.f - orifrac : orifrac;
            // Trilinear interpolation:
            h[at(y, x, o)] += wy * wx * wo * weight * mag;
          }
        }
      }
    }

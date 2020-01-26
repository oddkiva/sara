#include <math_constants.h>

#include <DO/Shakti/MultiArray/Grid.hpp>
#include <DO/Shakti/MultiArray/Offset.hpp>

#include <DO/Shakti/Segmentation/SuperPixel.hpp>


using namespace std;


namespace DO { namespace Shakti {

  // Dimensions of the image 2D array.
  __constant__ Vector2i image_sizes;
  __constant__ int image_padded_width;

  // Dimensions of the clusters 2D array
  __constant__ Vector2i num_clusters;
  __constant__ Vector2i cluster_sizes;
  __constant__ int clusters_padded_width;

  // Dimensions of the labels 2D array
  __constant__ int labels_padded_width;

  // Control parameter for the squared distance.
  __constant__ float distance_weight;

  __device__
  inline float squared_distance(const Vector2f& x1, const Vector4f& I1,
                                const Vector2f& x2, const Vector4f& I2)
  {
    return (I1 - I2).squared_norm() + distance_weight * (x1 - x2).squared_norm();
  }

  // At the beginning a cluster corresponds to a block in a grid.
  __global__
  void init_clusters(Cluster *out_clusters, const Vector4f *in_image)
  {
    const auto i_b = offset<2>();
    const auto p_b = coords<2>();

    // The image coordinates of the top-left corner of the block is:
    const auto tl_b = Vector2f{
      p_b.x()*cluster_sizes.x(),
      p_b.y()*cluster_sizes.y()
    };

    // The 2D image coordinates of the block center is:
    const Vector2f c_b = tl_b + Vector2f{ cluster_sizes.x() / 2, cluster_sizes.y() / 2 };
    // The image offset of the block center is:
    const int o_c_b = c_b.x() + c_b.y() * image_padded_width;

    out_clusters[i_b].num_points = 0;
    out_clusters[i_b].center = c_b;
    out_clusters[i_b].color = in_image[o_c_b];
  }

  __global__
  void assign_means(int *out_labels,
                    const Cluster *in_clusters, const Vector4f *in_image)
  {
    // For each thread in the block, populate the list of nearest cluster centers.
    __shared__ int x1, y1, x2, y2;
    __shared__ Vector4f I_c[3][3]; // I(c) = color value of a cluster center $c$.
    __shared__ Vector2f p_c[3][3]; // p(c) = coordinates of cluster center $c$.

    // Get the pixel 2D coordinates
    const auto _p_i = coords<2>();
    // Stop the kernel if we process the padded region of the image.
    if (_p_i.x() >= image_sizes.x() || _p_i.y() >= image_sizes.y())
      return;
    // Get the offset and color value.
    const auto p_i = Vector2f{ _p_i.x(), _p_i.y() };
    const auto i = offset<2>();
    const auto I_i = in_image[_p_i.x() + _p_i.y()*image_padded_width];

    const int x_t = threadIdx.x;
    const int y_t = threadIdx.y;

    // For each pixel (x, y), find the spatially nearest cluster centers.
    // In each block, the 3x3 top-left threads will populate the list of the cluster centers.
    // The other threads will wait.
    if (x_t < 3 && y_t < 3)
    {
      x1 = blockIdx.x == 0 ? 1 : 0;
      y1 = blockIdx.y == 0 ? 1 : 0;
      x2 = blockIdx.x >= num_clusters.x() ? 2 : 3;
      y2 = blockIdx.y >= num_clusters.y() ? 2 : 3;

      auto x_b = int(blockIdx.x) + x_t - 1;
      auto y_b = int(blockIdx.y) + y_t - 1;

      if ( 0 <= x_b && x_b < num_clusters.x() &&
           0 <= y_b && y_b < num_clusters.y() )
      {
        I_c[x_t][y_t] = in_clusters[x_b + y_b * clusters_padded_width].color;
        p_c[x_t][y_t] = in_clusters[x_b + y_b * clusters_padded_width].center;
      }
    }

    __syncthreads();

    // Assign the index of the nearest cluster to pixel (x,y).
    Vector2i nearest_cluster_idx{};
    float nearest_distance{ CUDART_INF_F };
#pragma unroll
    for (int x = x1; x < x2; ++x)
    {
#pragma unroll
      for (int y = y1; y < y2; ++y)
      {
        auto d = squared_distance(p_c[x][y], I_c[x][y], p_i, I_i);
        if (d < nearest_distance)
        {
          nearest_distance = d;
          nearest_cluster_idx = Vector2i{
            blockIdx.x + x - 1,
            blockIdx.y + y - 1
          };
        }
      }
    }

    const auto cluster_offset =
      nearest_cluster_idx.x() + nearest_cluster_idx.y() * num_clusters.x();
    out_labels[i] = cluster_offset;
  }

  __global__
  void update_means(Cluster *out_clusters,
                    const int *in_labels, const Vector4f *in_image)
  {
    // Be very careful with vicious issues due to memory padding!
    const auto cluster_offset = offset<2>();
    const auto cluster_pos = coords<2>();
    const auto cluster_idx = cluster_pos.x() + num_clusters.x() * cluster_pos.y();

    Vector2f cur_mean_pos = out_clusters[cluster_offset].center;
    Vector2f new_mean_pos{ Vector2f::Zero() };
    Vector4f new_mean_color{ Vector4f::Zero() };
    int num_points = 0;

    const int h_b = cluster_sizes.y();
    const int w_b = cluster_sizes.x();

    const int x1 = 0 <= cur_mean_pos.x() - w_b ? cur_mean_pos.x() - w_b : 0;
    const int y1 = 0 <= cur_mean_pos.y() - h_b ? cur_mean_pos.y() - h_b : 0;

    const int x2 = image_sizes.x() >= cur_mean_pos.x() + w_b?
                   cur_mean_pos.x() + w_b : image_sizes.x();

    const int y2 = image_sizes.y() >= cur_mean_pos.y() + h_b?
                   cur_mean_pos.y() + h_b : image_sizes.y();

    // Update to cluster centers.
    for (int y = y1; y < y2; ++y)
    {
      for (int x = x1; x < x2; ++x)
      {
        const auto image_offset = x + y * image_padded_width;
        const auto label_offset = x + y * labels_padded_width;

        Vector4f color = in_image[image_offset];
        int label = in_labels[label_offset];

        if (label == cluster_idx)
        {
          new_mean_color += color;
          new_mean_pos += Vector2f(x, y);
          ++num_points;
        }
      }
    }

    if (num_points == 0)
      return;

    new_mean_color /= float(num_points);
    new_mean_pos /= float(num_points);

    out_clusters[cluster_offset].color = new_mean_color;
    out_clusters[cluster_offset].center = new_mean_pos;
    out_clusters[cluster_offset].num_points = num_points;
  }

} /* namespace Shakti */
} /* namespace DO */


namespace DO { namespace Shakti {

  SegmentationSLIC::SegmentationSLIC()
  {
    _cluster_sizes = Vector2i{ 16, 16 };
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      cluster_sizes, &_cluster_sizes, sizeof(Vector2i)));

    set_distance_weight(0.f);
  }

  Vector2i SegmentationSLIC::get_image_sizes() const
  {
    auto sizes = Vector2i{};
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(
      &sizes, image_sizes, sizeof(Vector2i)));
    return sizes;
  }

  int SegmentationSLIC::get_image_padded_width() const
  {
    auto padded_width = int{};
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(
      &padded_width, image_padded_width, sizeof(int)));
    return padded_width;
  }

  float SegmentationSLIC::get_distance_weight() const
  {
    auto distance_weight = float{};
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyFromSymbol(
      &distance_weight, Shakti::distance_weight, sizeof(float)));
    return distance_weight;
  }

  void SegmentationSLIC::set_image_sizes(const Vector2i& sizes, int padded_width)
  {
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      image_sizes, &sizes, sizeof(Vector2i)));

    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      image_padded_width, &padded_width, sizeof(int)));

    // Set the image grid for CUDA.
    _image_block_sizes = default_block_size_2d();
    _image_grid_sizes = grid_size_2d(sizes, padded_width, _image_block_sizes);
  }

  void SegmentationSLIC::set_image_sizes(const MultiArray<Vector4f, 2>& device_image)
  {
    set_image_sizes(device_image.sizes(), device_image.padded_width());
  }

  void SegmentationSLIC::set_distance_weight(float distance_weight)
  {
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      Shakti::distance_weight, &distance_weight, sizeof(int)));
  }

  MultiArray<int, 2>
  SegmentationSLIC::operator()(const MultiArray<Vector4f, 2>& image)
  {
    set_image_sizes(image);

    MultiArray<int, 2> labels{ init_labels(image) };
    MultiArray<Cluster, 2> clusters{ init_clusters(image) };

    for (int i = 0; i < 5; ++i)
    {
      assign_means(labels, clusters, image);
      update_means(clusters, labels, image);
    }
    assign_means(labels, clusters, image);

    return labels;
  }

  void
  SegmentationSLIC::operator()(int *labels, const Vector4f *rgba_image, const int *sizes)
  {
    auto image_array = MultiArray<Vector4f, 2>{ rgba_image, sizes };
    auto labels_array = (*this)(image_array);
    labels_array.copy_to_host(labels);
  }

  MultiArray<Cluster, 2>
  SegmentationSLIC::init_clusters(const MultiArray<Vector4f, 2>& image)
  {
    // Compute the number of clusters.
    const auto& sizes = image.sizes();
    const auto num_clusters = Vector2i(
      // Number of clusters per columns.
      (sizes.x() + _image_block_sizes.x - 1) / _image_block_sizes.x,
      // Number of clusters per rows.
      (sizes.y() + _image_block_sizes.y - 1) / _image_block_sizes.y
    );

    // Initialize the clusters with the block centers.
    MultiArray<Cluster, 2> clusters{ num_clusters };

    // Register the dimensions of the clusters 2D array for CUDA.
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      Shakti::num_clusters, &num_clusters, sizeof(Vector2i)));
    const auto cluster_padded_width = clusters.padded_width();
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      Shakti::clusters_padded_width, &cluster_padded_width, sizeof(int)));

    // Set the cluster grid for CUDA.
    _cluster_block_sizes = dim3(_cluster_sizes.x(), _cluster_sizes.y());
    _cluster_grid_sizes = grid_size_2d(clusters, _cluster_block_sizes);
    Shakti::init_clusters<<<_cluster_grid_sizes, _cluster_block_sizes>>>(
      clusters.data(), image.data());

    return clusters;
  }

  MultiArray<int, 2>
  SegmentationSLIC::init_labels(const MultiArray<Vector4f, 2>& image)
  {
    // Initialize the labels with the block centers.
    MultiArray<int, 2> labels{ image.sizes() };

    // Register the dimensions of the labels 2D array for CUDA.
    const auto labels_padded_width = labels.padded_width();
    SHAKTI_SAFE_CUDA_CALL(cudaMemcpyToSymbol(
      Shakti::labels_padded_width, &labels_padded_width, sizeof(int)));

    // Set the labels grid for CUDA.
    _labels_block_sizes = dim3(_cluster_sizes.x(), _cluster_sizes.y());
    _labels_grid_sizes = grid_size_2d(labels);

    return labels;
  }

  void
  SegmentationSLIC::assign_means(MultiArray<int, 2>& labels,
                                 const MultiArray<Cluster, 2>& clusters,
                                 const MultiArray<Vector4f, 2>& image)
  {
    Shakti::assign_means<<<_labels_grid_sizes, _labels_block_sizes>>>(
      labels.data(), clusters.data(), image.data());
  }

  void
  SegmentationSLIC::update_means(MultiArray<Cluster, 2>& clusters,
                                 const MultiArray<int, 2>& labels,
                                 const MultiArray<Vector4f, 2>& image)
  {
    Shakti::update_means<<<_cluster_grid_sizes, _cluster_block_sizes>>>(
      clusters.data(), labels.data(), image.data());
  }

} /* namespace Shakti */
} /* namespace DO */
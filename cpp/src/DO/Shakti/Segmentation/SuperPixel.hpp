#ifndef DO_SHAKTI_SEGMENTATION_SUPERPIXEL_HPP
#define DO_SHAKTI_SEGMENTATION_SUPERPIXEL_HPP

#include <memory>
#include <tuple>

#include <DO/Shakti/Defines.hpp>

#include <DO/Shakti/MultiArray/Matrix.hpp>
#include <DO/Shakti/MultiArray/MultiArray.hpp>


#if defined(__CUDACC__) // NVCC
# define SHAKTI_CUDA_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
# define SHAKTI_CUDA_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
# define SHAKTI_CUDA_ALIGN(n) __declspec(align(n))
#else
# error "Please provide a definition for SHAKTI_CUDA_ALIGN macro for your host compiler!"
#endif


namespace DO { namespace Shakti {

  struct SHAKTI_CUDA_ALIGN(8) Cluster
  {
    Vector4f color;
    Vector2f center;
    int num_points;
  };

  class DO_SHAKTI_EXPORT SegmentationSLIC
  {
  public:
    //! @brief Constructor.
    SegmentationSLIC();

    //! @{
    //! @brief Getters.
    Vector2i get_image_sizes() const;

    int get_image_padded_width() const;

    float get_distance_weight() const;
    //! @}

    //! @{
    //! @brief Setters.
    void set_image_sizes(const Vector2i& sizes, int padded_width);

    void set_image_sizes(const MultiArray<Vector4f, 2>& device_image);

    void set_distance_weight(float weight);
    //! @}

  public:
    //! @{
    //! @brief Run the algorithm.
    MultiArray<int, 2> operator()(const MultiArray<Vector4f, 2>& image);

    void operator()(int *out_labels, const Vector4f *rgba_image, const int *sizes);
    //! @}

  public:
    MultiArray<Cluster, 2>
    init_clusters(const MultiArray<Vector4f, 2>& image);

    MultiArray<int, 2>
    init_labels(const MultiArray<Vector4f, 2>& image);

    void assign_means(MultiArray<int, 2>& labels,
                      const MultiArray<Cluster, 2>& clusters,
                      const MultiArray<Vector4f, 2>& image);

    void update_means(MultiArray<Cluster, 2>& clusters,
                      const MultiArray<int, 2>& labels,
                      const MultiArray<Vector4f, 2>& image);

  private:
    dim3 _image_block_sizes;
    dim3 _image_grid_sizes;

    Vector2i _cluster_sizes;
    dim3 _cluster_block_sizes;
    dim3 _cluster_grid_sizes;

    dim3 _labels_block_sizes;
    dim3 _labels_grid_sizes;
  };

} /* namespace Shakti */
} /* namespace DO */


#endif /* DO_SHAKTI_SEGMENTATION_SUPERPIXEL_HPP */

/**
 *  General definitions (header)
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date February, 2012
 */

#ifndef DEFS_H
#define DEFS_H

//== DEFINITIONS ===============================================================

#define WS 32 // Warp size (defines b x b block size where b = WS)

//== NAMESPACES ===============================================================

namespace gpufilter {

//== PROTOTYPES ===============================================================

/**
 *  @ingroup utils
 *  @brief Calculate image borders
 *
 *  Given the image size and an extension to consider calculates the
 *  four-directional border pixels (left, top, right and bottom)
 *  needed by algorithm 4 and 5.
 *
 *  @param[out] left Extension (in pixels) in the left border of the image
 *  @param[out] top Extension (in pixels) in the top border of the image
 *  @param[out] right Extension (in pixels) in the right border of the image
 *  @param[out] bottom Extension (in pixels) in the bottom border of the image
 *  @param[in] w Width of the image
 *  @param[in] h Height of the image
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 */
extern
void calc_borders( int& left,
                   int& top,
                   int& right,
                   int& bottom,
                   const int& w,
                   const int& h,
                   const int& extb );

/**
 *  @ingroup utils
 *  @brief Verify if an image needs to be extended
 *
 *  Given the image size and an extension to consider verifies if the
 *  image should be extended to met the requirement of each dimension
 *  be a multiple of 32.
 *
 *  @param[in] w Width of the image
 *  @param[in] h Height of the image
 *  @param[in] extb Extension (in blocks) to consider outside image (default 0)
 *  @return True if corresponding image should be extended
 */
extern
bool extend( const int& w,
             const int& h,
             const int& extb );

//=============================================================================
} // namespace gpufilter
//=============================================================================
#endif // DEFS_H
//=============================================================================

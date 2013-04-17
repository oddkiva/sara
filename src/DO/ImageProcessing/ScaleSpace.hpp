/*
 * =============================================================================
 *
 *       Filename:  ScaleSpace.hpp
 *
 *    Description:  TODO: Determinant of hessians.
 *                  Calcul d'image integrale.
 *
 *        Version:  1.0
 *        Created:  16/07/2011 10:30:30
 *       Revision:  none
 *       Compiler:  msvc
 *
 *         Author:  David OK (DO), david.ok@imagine.enpc.fr 
 *        Company:  IMAGINE, (Ecole des Ponts ParisTech & CSTB)
 *
 * =============================================================================
 */

#ifndef DO_IMAGEPROCESSING_SCALESPACE_HPP
#define DO_IMAGEPROCESSING_SCALESPACE_HPP

namespace DO {
  
  struct ScaleInfo
  {
    ScaleInfo(float scale_, float octave_, float octave_size_)
      : scale(scale_), octave(octave_), octaveSize(octave_size_) {}
    float scale;
    float octave;
    float octaveSize;
  };
  
  class ComputeScaleSpace 
  {
  public:
    inline ComputeScaleSpace(int firstOctave = -1,
                             int numScalesPerOctave = 5,
                             float initialSigma = 1.6f,
                             int borderSize = 5)
      : first_octave_(firstOctave)
      , num_scales_per_octave_(numScalesPerOctave)
      , initial_sigma_(initialSigma)
      , border_size_(borderSize) {}

    void operator()(std::vector<Image<float> >& gaussians,
                    std::vector<ScaleInfo>& scales,
                    const Image<float>& image);

    void checkParams() const;

  private:
    int first_octave_;
    int num_scales_per_octave_;
    float initial_sigma_;
    int border_size_;
  };

  void computeDoGs(std::vector<Image<float> >& dogs,
                   const std::vector<Image<float> >& gaussians);

  void computeDoHs(std::vector<Image<Matrix2f> >& hessians,
                   const std::vector<Image<float> >& gausians);

} /* namespace DO */

#endif /* DO_IMAGEPROCESSING_SCALESPACE_HPP */

// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <DO/FeatureDescriptors.hpp>
#include <daisy/daisy.h>

namespace DO {

  DAISY::DAISY() : daisy_computer_(new daisy())
  {
    int rad   = 15;
    int radq  =  3;
    int thq   =  8;
    int histq =  8;
    int orientation_resolution = 18;
    int nrm_type = NRM_PARTIAL;
    bool disable_interpolation = false;

    daisy_computer_->verbose(0);
    
    if( disable_interpolation )
      daisy_computer_->disable_interpolation();
    daisy_computer_->set_parameters(rad, radq, thq, histq);
    if( nrm_type == 0 ) daisy_computer_->set_normalization( NRM_PARTIAL );
    if( nrm_type == 1 ) daisy_computer_->set_normalization( NRM_FULL );
    if( nrm_type == 2 ) daisy_computer_->set_normalization( NRM_SIFT );
  }

  DAISY::~DAISY()
  {
    if (daisy_computer_)
      delete daisy_computer_;
  }

  void DAISY::initialize(const Image<float>& image) const
  {
    daisy_computer_->set_image(
      const_cast<float *>(image.data()),
      image.height(), image.width() );
    daisy_computer_->initialize_single_descriptor_mode();
  }

  void DAISY::reset() const
  {
    daisy_computer_->reset();
  }

  int DAISY::dimension() const { return daisy_computer_->descriptor_size(); }

  void DAISY::compute(VectorXf& desc, float x, float y, float o) const
  {
    if (desc.size() != daisy_computer_->descriptor_size())
      desc.resize(daisy_computer_->descriptor_size());
    //CHECK(desc.size());

    daisy_computer_->get_descriptor(y, x, o, desc.data());
    /*kutility::display(
      desc.data(),
      daisy_computer_->grid_point_number(),
      daisy_computer_->get_hq(), 0, 0 );*/
  }
  
  void DAISY::compute(DescriptorMatrix<float>& daisies,
                      const std::vector<OERegion>& features,
                      const std::vector<Vector2i>& scaleOctPairs,
                      const ImagePyramid<float>& pyramid) const
  {
    VectorXf daisy_descriptor;
    Vector2i currentScaleOct(-1,-1);
    daisies.resize(features.size(), daisy_computer_->descriptor_size());
    for (int i = 0; i != features.size(); ++i)
    {
      if (currentScaleOct != scaleOctPairs[i])
      {
        currentScaleOct = scaleOctPairs[i];
        //CHECK(currentScaleOct.transpose());
        int s = scaleOctPairs[i](0);
        int o = scaleOctPairs[i](1);

        if (currentScaleOct != scaleOctPairs[0])
        {
          //printStage("Resetting");
          reset();
        }
        //printStage("Initializing");
        initialize(pyramid(s,o));
        //waitReturnKey();
      }

      const OERegion& f = features[i];
      float ori = f.orientation() < 0 ? f.orientation()+float(M_PI) : f.orientation();
      ori = ori / float(M_PI) * 180.f;
      if (ori < 0 && ori >= 360)
      {
        CHECK(ori);
        std::cerr << "Error wrong orientation" << std::endl;
      }
      int i_ori = ori >= 360 ? int(ori) : 0;
      compute(daisy_descriptor, f.x(), f.y(), i_ori);

      daisies[i] = daisy_descriptor;

      /*cout << daisies[i].transpose() << endl;
      waitReturnKey();*/
    }
  }

} /* namespace DO */
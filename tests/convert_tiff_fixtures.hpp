#ifndef _CONVERT_TIFF_FIXTURES_H_
#define _CONVERT_TIFF_FIXTURES_H_
#include <iostream> 
#include <iomanip> 
#include <vector>
#include <cmath>
#include <stdexcept>
#include "boost/multi_array.hpp"
#include <boost/filesystem.hpp>

#include <string>
#include <sstream>
#include "image_stack_utils.h"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"

////////////////////////////////////////////////////////////////////////////

namespace multiviewnative {
 
  void fill(const ViewFromDisk& _in, view_data& _value){

    _value.image_   = (imageType*)_in.image()  ->data();
    _value.kernel1_ = (imageType*)_in.kernel1()->data();
    _value.kernel2_ = (imageType*)_in.kernel2()->data();
    _value.weights_ = (imageType*)_in.weights()->data();
   
    _value.image_dims_   = (int*)_in.image_dims();
    _value.kernel1_dims_ = (int*)_in.kernel1_dims();
    _value.kernel2_dims_ = (int*)_in.kernel2_dims();
    _value.weights_dims_ = (int*)_in.image_dims();

  }
  

  void fill_workspace(const ReferenceData& _ref, workspace& _space, const double& _lambda, const float& _minValue){
    _space.num_views = _ref.views_.size();

    if(_space.data_ != 0)
      delete [] _space.data_;

    _space.data_ = new view_data[_space.num_views];

    for(unsigned num = 0;num<_space.num_views;++num){
      fill(_ref.views_[num], _space.data_[num]);
    }

    _space.lambda_ = _lambda;
    _space.minValue_ = _minValue;

  }

  

}

#endif

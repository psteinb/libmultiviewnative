#ifndef _TIFF_FIXTURES_H_
#define _TIFF_FIXTURES_H_
#include <iostream> 
#include <iomanip> 
#include <vector>
#include <cmath>
//#include "mxn_indexer.hpp"
#include <boost/static_assert.hpp>
#include "boost/multi_array.hpp"

#include <string>
#include <sstream>
#include "image_stack_utils.h"
#include "tiff_utils.h"

////////////////////////////////////////////////////////////////////////////
// Explanation of the test images
// input :
// image_view_i.tif	.. the input frame stack from view i
// kernel1_view_i.tif	.. point spread function 1 from view i
// kernel2_view_i.tif	.. point spread function 2 from view i
// weights_view_i.tif	.. the weights from view i
// results:
// psi_i.tif		.. the results after the i-th iteration
// psi_0		.. first guess (all pixels have the same intensity)

namespace multiviewnative {

  static const std::string path_to_test_images = "/home/steinbac/development/libmultiview_data/";

  template <int ViewNumber = 0>
  struct ReferenceDataLoader
  {

    std::stringstream  image_path_  ;
    std::stringstream  kernel1_path_   ;
    std::stringstream  kernel2_path_   ;
    std::stringstream  weights_path_   ;

    TIFF*  image_tiff_  ;
    TIFF*  kernel1_tiff_   ;
    TIFF*  kernel2_tiff_   ;
    TIFF*  weights_tiff_   ;

    image_stack  image_     ;
    image_stack  kernel1_   ;
    image_stack  kernel2_   ;
    image_stack  weights_   ;

    // BOOST_STATIC_ASSERT(path_to_test_images.empty() != true);
    BOOST_STATIC_ASSERT(ViewNumber >= 0 && ViewNumber < 6);

    ReferenceDataLoader():
      image_path_    (  ""  )  ,
      kernel1_path_  (  ""  )  ,
      kernel2_path_  (  ""  )  ,
      weights_path_  (  ""  )  ,
      image_tiff_    (  0   )  ,
      kernel1_tiff_  (  0   )  ,
      kernel2_tiff_  (  0   )  ,
      weights_tiff_  (  0   )  ,
      image_         (     )  ,
      kernel1_       (     )  ,
      kernel2_       (     )  ,
      weights_       (     )
    {
      image_path_    << path_to_test_images <<  "image_view_"    <<  ViewNumber  <<  ".tif";
      kernel1_path_  << path_to_test_images <<  "kernel1_view_"  <<  ViewNumber  <<  ".tif";
      kernel2_path_  << path_to_test_images <<  "kernel2_view_"  <<  ViewNumber  <<  ".tif";
      weights_path_  << path_to_test_images <<  "weights_view_"  <<  ViewNumber  <<  ".tif";
    
      image_tiff_   = TIFFOpen( image_path_  .str().c_str() , "r" );
      kernel1_tiff_ = TIFFOpen( kernel1_path_.str().c_str() , "r" );
      kernel2_tiff_ = TIFFOpen( kernel2_path_.str().c_str() , "r" );
      weights_tiff_ = TIFFOpen( weights_path_.str().c_str() , "r" );

      std::vector<tdir_t> image_tdirs   ;get_tiff_dirs(image_tiff_,   image_tdirs  );
      std::vector<tdir_t> kernel1_tdirs ;get_tiff_dirs(kernel1_tiff_, kernel1_tdirs);
      std::vector<tdir_t> kernel2_tdirs ;get_tiff_dirs(kernel2_tiff_, kernel2_tdirs);
      std::vector<tdir_t> weights_tdirs ;get_tiff_dirs(weights_tiff_, weights_tdirs);

      extract_tiff_to_image_stack(image_tiff_,   image_tdirs  , image_     );
      extract_tiff_to_image_stack(kernel1_tiff_, kernel1_tdirs, kernel1_   );
      extract_tiff_to_image_stack(kernel2_tiff_, kernel2_tdirs, kernel2_   );
      extract_tiff_to_image_stack(weights_tiff_, weights_tdirs, weights_   );

    }

    virtual ~ReferenceDataLoader()  { 


      if(image_tiff_)
	TIFFClose( image_tiff_ );

      if(kernel1_tiff_)
	TIFFClose( kernel1_tiff_ );

      if(kernel2_tiff_)
	TIFFClose( kernel2_tiff_ );

      if(weights_tiff_)
	TIFFClose( weights_tiff_ );

    };

  

  };

  typedef ReferenceDataLoader<0> view0_loader;
  typedef ReferenceDataLoader<1> view1_loader;
  typedef ReferenceDataLoader<2> view2_loader;
  typedef ReferenceDataLoader<3> view3_loader;
  typedef ReferenceDataLoader<4> view4_loader;
  typedef ReferenceDataLoader<5> view5_loader;

  template<int ViewNumber>
  struct DeconvolutionFixture
  {
    image_stack  image_     ;
    image_stack  kernel1_   ;
    image_stack  kernel2_   ;
    image_stack  weights_   ;

    DeconvolutionFixture():
      image_(),
      kernel1_(),
      kernel2_(),
      weights_(){}

    DeconvolutionFixture(const ReferenceDataLoader<ViewNumber>& _other):
      image_(_other.image_),
      kernel1_(_other.kernel1_),
      kernel2_(_other.kernel2_),
      weights_(_other.weights_){}

    DeconvolutionFixture& operator=(const ReferenceDataLoader<ViewNumber>& _other){
      image_    =  _other.image_    ;
      kernel1_  =  _other.kernel1_  ;
      kernel2_  =  _other.kernel2_  ;
      weights_  =  _other.weights_  ;
    }

    void setup_from(const ReferenceDataLoader<ViewNumber>& _other){
      this->image_   .resize( boost::extents[_other.image_  .shape()[0]][_other.image_  .shape()[1]][_other.image_  .shape()[2]] )  ;
      this->kernel1_ .resize( boost::extents[_other.kernel1_.shape()[0]][_other.kernel1_.shape()[1]][_other.kernel1_.shape()[2]] )  ;
      this->kernel2_ .resize( boost::extents[_other.kernel2_.shape()[0]][_other.kernel2_.shape()[1]][_other.kernel2_.shape()[2]] )  ;
      this->weights_ .resize( boost::extents[_other.weights_.shape()[0]][_other.weights_.shape()[1]][_other.weights_.shape()[2]] )  ;
      this->image_    =  _other.image_    ;
      this->kernel1_  =  _other.kernel1_  ;
      this->kernel2_  =  _other.kernel2_  ;
      this->weights_  =  _other.weights_  ;
    }
  };

  typedef DeconvolutionFixture<0> view0_fixture;
  typedef DeconvolutionFixture<1> view1_fixture;
  typedef DeconvolutionFixture<2> view2_fixture;
  typedef DeconvolutionFixture<3> view3_fixture;
  typedef DeconvolutionFixture<4> view4_fixture;
  typedef DeconvolutionFixture<5> view5_fixture;
}

#endif

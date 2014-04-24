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

  static const std::string path_to_test_images = "/dev/shm/libmultiview_data/";
  
  struct ViewFromDisk
  {

    int view_number_;

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
    
    ViewFromDisk(const int& _view_number = -1):
      view_number_ ( _view_number ) ,
      image_path_             (  ""  )  ,
      kernel1_path_           (  ""  )  ,
      kernel2_path_           (  ""  )  ,
      weights_path_           (  ""  )  ,
      image_tiff_             (  0   )  ,
      kernel1_tiff_           (  0   )  ,
      kernel2_tiff_           (  0   )  ,
      weights_tiff_           (  0   )  ,
      image_                  (  )   ,
      kernel1_                (  )   ,
      kernel2_                (  )   ,
      weights_                (  )   

    {
      if(view_number_>-1)
	this->load(view_number_);
    }

    ViewFromDisk(const ViewFromDisk& _rhs):
      view_number_               (              _rhs.view_number_   )                                 ,
      image_path_                (              _rhs.image_path_.str()    )                                 ,
      kernel1_path_              (              _rhs.kernel1_path_.str()  )                                 ,
      kernel2_path_              (              _rhs.kernel2_path_.str()  )                                 ,
      weights_path_              (              _rhs.weights_path_.str()  )                                 ,
      image_tiff_                (              TIFFOpen(           _rhs.image_path_  .str().c_str()  ,    "r"  )  )  ,
      kernel1_tiff_              (              TIFFOpen(           _rhs.kernel1_path_.str().c_str()  ,    "r"  )  )  ,
      kernel2_tiff_              (              TIFFOpen(           _rhs.kernel2_path_.str().c_str()  ,    "r"  )  )  ,
      weights_tiff_              (              TIFFOpen(           _rhs.weights_path_.str().c_str()  ,    "r"  )  )  ,
      image_                     (              _rhs.image_         )                                 ,
      kernel1_                   (              _rhs.kernel1_       )                                 ,
      kernel2_                   (              _rhs.kernel2_       )                                 ,
      weights_                   (              _rhs.weights_       )
    {
      if(view_number_>-1)
	this->load(view_number_);
    }

    ViewFromDisk& operator=(const ViewFromDisk& _rhs){
 
      this->clear();

      view_number_   =  _rhs.view_number_   ;
      image_path_  .str(_rhs.image_path_  .str())  ;
      kernel1_path_.str(_rhs.kernel1_path_.str())  ;
      kernel2_path_.str(_rhs.kernel2_path_.str())  ;
      weights_path_.str(_rhs.weights_path_.str())  ;

      image_tiff_   = TIFFOpen( image_path_  .str().c_str() , "r" );
      kernel1_tiff_ = TIFFOpen( kernel1_path_.str().c_str() , "r" );
      kernel2_tiff_ = TIFFOpen( kernel2_path_.str().c_str() , "r" );
      weights_tiff_ = TIFFOpen( weights_path_.str().c_str() , "r" );

      image_         =  _rhs.image_         ;
      kernel1_       =  _rhs.kernel1_       ;
      kernel2_       =  _rhs.kernel2_       ;
      weights_       =  _rhs.weights_       ;

      return *this;
	    
    }
    
    ViewFromDisk& operator=(const int& _view_number){

      this->clear();

      view_number_ = _view_number;
      
      if(view_number_>-1)
	this->load(view_number_);
      
      return *this;
    }

    void load(const int& _view_number){

      view_number_ = _view_number;

      image_path_    << path_to_test_images <<  "image_view_"    <<  view_number_  <<  ".tif";
      kernel1_path_  << path_to_test_images <<  "kernel1_view_"  <<  view_number_  <<  ".tif";
      kernel2_path_  << path_to_test_images <<  "kernel2_view_"  <<  view_number_  <<  ".tif";
      weights_path_  << path_to_test_images <<  "weights_view_"  <<  view_number_  <<  ".tif";
    
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

    void load(const ViewFromDisk& _rhs){
      
      this->clear();

      view_number_   =  _rhs.view_number_   ;
      image_path_  .str(_rhs.image_path_  .str())  ;
      kernel1_path_.str(_rhs.kernel1_path_.str())  ;
      kernel2_path_.str(_rhs.kernel2_path_.str())  ;
      weights_path_.str(_rhs.weights_path_.str())  ;

      image_tiff_   = TIFFOpen( image_path_  .str().c_str() , "r" );
      kernel1_tiff_ = TIFFOpen( kernel1_path_.str().c_str() , "r" );
      kernel2_tiff_ = TIFFOpen( kernel2_path_.str().c_str() , "r" );
      weights_tiff_ = TIFFOpen( weights_path_.str().c_str() , "r" );
      
      std::vector<unsigned> image_shape(3);
      std::copy(&_rhs.image_  .shape()[0], &_rhs.image_  .shape()[0] + 3,image_shape.begin());

      std::vector<unsigned> kernel1_shape(3);
      std::copy(&_rhs.kernel1_  .shape()[0], &_rhs.kernel1_  .shape()[0] + 3,kernel1_shape.begin());

      std::vector<unsigned> kernel2_shape(3);
      std::copy(&_rhs.kernel2_  .shape()[0], &_rhs.kernel2_  .shape()[0] + 3,kernel2_shape.begin());

      std::vector<unsigned> weights_shape(3);
      std::copy(&_rhs.weights_  .shape()[0], &_rhs.weights_  .shape()[0] + 3,weights_shape.begin());

      image_         .resize(  image_shape )       ;
      kernel1_       .resize(  kernel1_shape )       ;
      kernel2_       .resize(  kernel2_shape )       ;
      weights_       .resize(  weights_shape )       ;

      image_         =  _rhs.image_         ;
      kernel1_       =  _rhs.kernel1_       ;
      kernel2_       =  _rhs.kernel2_       ;
      weights_       =  _rhs.weights_       ;

    }


    void clear(){


      if(image_tiff_)
	TIFFClose( image_tiff_ );

      if(kernel1_tiff_)
	TIFFClose( kernel1_tiff_ );

      if(kernel2_tiff_)
	TIFFClose( kernel2_tiff_ );

      if(weights_tiff_)
	TIFFClose( weights_tiff_ );

    }

    virtual ~ViewFromDisk()  { 

      clear();
      
    };

  

  };

  struct ReferenceData {
    
    std::vector<ViewFromDisk> views_;

    ReferenceData():
      views_(6){
      for(int i = 0;i<6;++i)
	views_[i].load(i);
    }

    ReferenceData(const ReferenceData& _rhs):
      views_(_rhs.views_){
    }

    void copy_in(const ReferenceData& _other){
      for(int i = 0;i<6;++i)
	views_[i] = _other.views_[i];
    }
  };
  
  }

#endif

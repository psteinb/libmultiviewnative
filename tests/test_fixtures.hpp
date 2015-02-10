#ifndef _TEST_FIXTURES_H_
#define _TEST_FIXTURES_H_
#include <iostream> 
#include <iomanip> 
#include <vector>
#include <cmath>
#include <numeric>
//#include "mxn_indexer.hpp"
#include <boost/static_assert.hpp>
#include "boost/multi_array.hpp"

//http://www.boost.org/doc/libs/1_55_0/libs/multi_array/doc/user.html
//http://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order
#include "image_stack_utils.h"
#include "test_algorithms.hpp"



namespace multiviewnative {

template <unsigned short KernelDimSize = 3, 
	  unsigned ImageDimSize = 8
	  >
struct convolutionFixture3D
{

  static const unsigned halfKernel  = KernelDimSize/2u;
  static const unsigned imageDimSize  = ImageDimSize;
  static const unsigned kernelDimSize  = KernelDimSize;

  const unsigned    image_size_   ;
  std::vector<int>  image_dims_                             ;
  std::vector<int>  padded_image_dims_                             ;
  image_stack       image_                                  ;
  image_stack       one_                                  ;
  image_stack       padded_image_                           ;
  image_stack       padded_one_                           ;
  image_stack       asymm_padded_image_                           ;
  image_stack       asymm_padded_one_                           ;
  image_stack       image_folded_by_horizontal_             ;
  image_stack       image_folded_by_vertical_               ;
  image_stack       image_folded_by_depth_                  ;
  image_stack       image_folded_by_all1_                   ;
  image_stack       one_folded_by_asymm_cross_kernel_     ;
  image_stack       one_folded_by_asymm_one_kernel_       ;
  image_stack       one_folded_by_asymm_identity_kernel_  ;

  const unsigned    kernel_size_  ;
  std::vector<int>  kernel_dims_                            ;
  std::vector<int>  asymm_kernel_dims_                            ;
  image_stack       trivial_kernel_                         ;
  image_stack       identity_kernel_                        ;
  image_stack       vertical_kernel_                        ;
  image_stack       horizont_kernel_                        ;
  image_stack       depth_kernel_                           ;
  image_stack       all1_kernel_                            ;
  image_stack       asymm_cross_kernel_                     ;
  image_stack       asymm_one_kernel_                       ;
  image_stack       asymm_identity_kernel_                  ;

  
  BOOST_STATIC_ASSERT(KernelDimSize % 2 != 0);

public:
  
  convolutionFixture3D():
    image_size_                             ((unsigned)std::pow(ImageDimSize,3)),
    image_dims_                             (3,ImageDimSize),
    padded_image_dims_                             (3,ImageDimSize+2*(KernelDimSize/2)),
    image_                                  (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    one_                                  (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    padded_image_                           (boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
    padded_one_                           (boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
    asymm_padded_image_                     (),
    asymm_padded_one_                     (),
    image_folded_by_horizontal_             (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    image_folded_by_vertical_               (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    image_folded_by_depth_                  (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    image_folded_by_all1_                   (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    one_folded_by_asymm_cross_kernel_     (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    one_folded_by_asymm_one_kernel_       (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    one_folded_by_asymm_identity_kernel_  (boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    kernel_size_                            ((unsigned)std::pow(KernelDimSize,3)),
    kernel_dims_                            (3,KernelDimSize),
    asymm_kernel_dims_                            (3,KernelDimSize),
    trivial_kernel_                         (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    identity_kernel_                        (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    vertical_kernel_                        (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    horizont_kernel_                        (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    depth_kernel_                           (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    all1_kernel_                            (boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    asymm_cross_kernel_                     (boost::extents[KernelDimSize+1][KernelDimSize][KernelDimSize-1]),
    asymm_one_kernel_                       (boost::extents[KernelDimSize+1][KernelDimSize][KernelDimSize-1]),
    asymm_identity_kernel_                  (boost::extents[KernelDimSize+1][KernelDimSize][KernelDimSize-1])
  {
    
    //FILL KERNELS
    
        
    std::fill(trivial_kernel_.data()       ,trivial_kernel_.data()       +  kernel_size_                           ,0.f);
    std::fill(identity_kernel_.data()      ,identity_kernel_.data()      +  kernel_size_                           ,0.f);
    std::fill(vertical_kernel_.data()      ,vertical_kernel_.data()      +  kernel_size_                           ,0.f);
    std::fill(depth_kernel_.data()         ,depth_kernel_.data()         +  kernel_size_                           ,0.f);
    std::fill(horizont_kernel_.data()      ,horizont_kernel_.data()      +  kernel_size_                           ,0.f);
    std::fill(all1_kernel_.data()          ,all1_kernel_.data()          +  kernel_size_                           ,1.f);
    std::fill(asymm_cross_kernel_.data()     ,asymm_cross_kernel_.data()     +  asymm_cross_kernel_.num_elements()     ,0.f);
    std::fill(asymm_one_kernel_.data()       ,asymm_one_kernel_.data()       +  asymm_one_kernel_.num_elements()       ,0.f);
    std::fill(asymm_identity_kernel_.data()  ,asymm_identity_kernel_.data()  +  asymm_identity_kernel_.num_elements()  ,0.f);


    identity_kernel_.data()[kernel_size_/2]=1.; 

    for(unsigned int index = 0;index<KernelDimSize;++index){
      horizont_kernel_[index][halfKernel][halfKernel] = float(index+1);
      vertical_kernel_[halfKernel][index][halfKernel] = float(index+1);
      depth_kernel_   [halfKernel][halfKernel][index] = float(index+1);
    }
    
    asymm_identity_kernel_[asymm_cross_kernel_.shape()[0]/2][asymm_cross_kernel_.shape()[1]/2][asymm_cross_kernel_.shape()[2]/2] = 1.f;
    for(int z_index = 0;z_index<int(asymm_cross_kernel_.shape()[2]);++z_index){
      for(int y_index = 0;y_index<int(asymm_cross_kernel_.shape()[1]);++y_index){
    	for(int x_index = 0;x_index<int(asymm_cross_kernel_.shape()[0]);++x_index){
	  
	  if(z_index == (int)asymm_cross_kernel_.shape()[2]/2 && y_index == (int)asymm_cross_kernel_.shape()[1]/2){
	    asymm_cross_kernel_[x_index][y_index][z_index] = x_index + 1;
	    asymm_one_kernel_[x_index][y_index][z_index] = 1;
	  }
	  
	  if(x_index == (int)asymm_cross_kernel_.shape()[0]/2 && y_index == (int)asymm_cross_kernel_.shape()[1]/2){
	    asymm_cross_kernel_[x_index][y_index][z_index] = z_index + 101;
	    asymm_one_kernel_[x_index][y_index][z_index] = 1;
	  }
	  
	  if(x_index == (int)asymm_cross_kernel_.shape()[0]/2 && z_index == (int)asymm_cross_kernel_.shape()[2]/2){
	    asymm_cross_kernel_[x_index][y_index][z_index] = y_index + 11;
	    asymm_one_kernel_[x_index][y_index][z_index] = 1;
	  }
	  
	  
    	}
      }
    }

    //FILL IMAGES
    unsigned padded_image_axis = ImageDimSize+2*halfKernel;
    unsigned padded_image_size = std::pow(padded_image_axis,3);
    std::fill(image_.data(),         image_.data()         +  image_size_,        0.f  );
    std::fill(one_.data(),         one_.data()         +  image_size_,        0.f  );
    std::fill(padded_image_.data(),  padded_image_.data()  +  padded_image_size,  0.f  );
    std::fill(padded_one_.data(),  padded_one_.data()  +  padded_image_size,  0.f  );
    padded_one_[padded_image_axis/2][padded_image_axis/2][padded_image_axis/2] = 1.f;
    one_[ImageDimSize/2][ImageDimSize/2][ImageDimSize/2] = 1.f;

    unsigned image_index=0;
    for(int z_index = 0;z_index<int(image_dims_[2]);++z_index){
      for(int y_index = 0;y_index<int(image_dims_[1]);++y_index){
    	for(int x_index = 0;x_index<int(image_dims_[0]);++x_index){
    	  image_index=x_index;
    	  image_index += y_index*image_dims_[0];
    	  image_index += z_index*image_dims_[0]*image_dims_[1] ;
    	  image_[x_index][y_index][z_index] = float(image_index);
    	}
      }
    }

    //PADD THE IMAGE FOR CONVOLUTION
    range axis_subrange = range(halfKernel,halfKernel+ImageDimSize);
    image_stack_view padded_image_original = padded_image_[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
    padded_image_original = image_;
    
    image_stack padded_image_folded_by_horizontal  = padded_image_;
    image_stack padded_image_folded_by_vertical    = padded_image_;
    image_stack padded_image_folded_by_depth       = padded_image_;
    image_stack padded_image_folded_by_all1        = padded_image_;

    //PREPARE ASYMM IMAGES
    std::vector<unsigned> symm_offsets(3);
    std::vector<unsigned> asymm_offsets(3);
    std::vector<unsigned> asymm_padded_image_shape(3);
    std::vector<range> asymm_axis_subrange(3);
    for(int i = 0;i<3;++i){
      asymm_kernel_dims_[i] = asymm_cross_kernel_.shape()[i];
      symm_offsets[i] = halfKernel;
      asymm_offsets[i] = asymm_cross_kernel_.shape()[i]/2;
      asymm_axis_subrange[i] = range(asymm_offsets[i],asymm_offsets[i]+ImageDimSize);
      asymm_padded_image_shape[i] = ImageDimSize+2*asymm_offsets[i];
    }

    asymm_padded_image_.resize(asymm_padded_image_shape)    ;
    asymm_padded_one_.resize(asymm_padded_image_shape)    ;
    std::fill(asymm_padded_image_.data(),         asymm_padded_image_.data()         +  asymm_padded_image_.num_elements(),        0.f  );
    std::fill(asymm_padded_one_.data(),         asymm_padded_one_.data()         +  asymm_padded_one_.num_elements(),        0.f  );
    asymm_padded_one_[asymm_padded_one_.shape()[0]/2][asymm_padded_one_.shape()[1]/2][asymm_padded_one_.shape()[2]/2] = 1.f;

    image_stack_view asymm_padded_image_original = asymm_padded_image_[ boost::indices[asymm_axis_subrange[0]][asymm_axis_subrange[1]][asymm_axis_subrange[2]] ];
    asymm_padded_image_original = image_;

    image_stack asymm_padded_one_folded_by_asymm_cross_kernel    = asymm_padded_one_;
    image_stack asymm_padded_one_folded_by_asymm_one_kernel      = asymm_padded_one_;
    image_stack asymm_padded_one_folded_by_asymm_identity_kernel = asymm_padded_one_;

    //CONVOLVE
    convolve(padded_image_, horizont_kernel_, padded_image_folded_by_horizontal, symm_offsets);
    convolve(padded_image_, vertical_kernel_, padded_image_folded_by_vertical, symm_offsets);
    convolve(padded_image_, depth_kernel_, padded_image_folded_by_depth, symm_offsets);
    convolve(padded_image_, all1_kernel_, padded_image_folded_by_all1, symm_offsets);

    convolve(asymm_padded_one_  ,  asymm_cross_kernel_     ,  asymm_padded_one_folded_by_asymm_cross_kernel     ,  asymm_offsets);
    convolve(asymm_padded_one_  ,  asymm_one_kernel_       ,  asymm_padded_one_folded_by_asymm_one_kernel       ,  asymm_offsets);
    convolve(asymm_padded_one_  ,  asymm_identity_kernel_  ,  asymm_padded_one_folded_by_asymm_identity_kernel  ,  asymm_offsets);
    
    // for(int z_index = halfKernel;z_index<int(padded_image_axis-halfKernel);++z_index){
    //   for(int y_index = halfKernel;y_index<int(padded_image_axis-halfKernel);++y_index){
    // 	for(int x_index = halfKernel;x_index<int(padded_image_axis-halfKernel);++x_index){
	  	  
    // 	  padded_image_folded_by_horizontal[x_index][y_index][z_index] = 0.f;
    // 	  padded_image_folded_by_vertical[x_index][y_index][z_index] = 0.f;
    // 	  padded_image_folded_by_depth[x_index][y_index][z_index] = 0.f;
    // 	  padded_image_folded_by_all1[x_index][y_index][z_index] = 0.f;

    // 	  for(int kindex = 0;kindex<KernelDimSize;++kindex){
    // 	    //convolution in x
    // 	    kernel_value  =  horizont_kernel_[KernelDimSize-1-kindex][halfKernel][halfKernel]	;
    // 	    image_value   =  padded_image_[x_index-halfKernel+kindex][y_index][z_index]		;
    // 	    padded_image_folded_by_horizontal[x_index][y_index][z_index] += kernel_value*image_value;

    // 	    //convolution in y
    // 	    kernel_value  = vertical_kernel_[halfKernel][KernelDimSize-1-kindex][halfKernel];
    // 	    image_value   = padded_image_[x_index][y_index-halfKernel+kindex][z_index];
    // 	    padded_image_folded_by_vertical[x_index][y_index][z_index] += kernel_value*image_value;
	      

    // 	    //convolution in z
    // 	    kernel_value  = depth_kernel_[halfKernel][halfKernel][KernelDimSize-1-kindex];
    // 	    image_value   = padded_image_[x_index][y_index][z_index-halfKernel+kindex];
    // 	    padded_image_folded_by_depth[x_index][y_index][z_index] += kernel_value*image_value;
	      
    // 	  }
  

    // 	  newValue = 0.;
    // 	  for(int z_kernel = -(int)halfKernel;z_kernel<=((int)halfKernel);++z_kernel){
    // 	    for(int y_kernel = -(int)halfKernel;y_kernel<=((int)halfKernel);++y_kernel){
    // 	      for(int x_kernel = -(int)halfKernel;x_kernel<=((int)halfKernel);++x_kernel){
    // 		newValue += padded_image_[x_index+x_kernel][y_index+y_kernel][z_index+z_kernel]*all1_kernel_[halfKernel+x_kernel][halfKernel+y_kernel][halfKernel+z_kernel];
    // 	      }
    // 	    }
    // 	  }
    // 	  padded_image_folded_by_all1[x_index][y_index][z_index] = newValue;
	
    // 	}
     
    //   }
    // }
    
    //EXTRACT NON-PADDED CONTENT FROM CONVOLVED IMAGE STACKS
    image_folded_by_horizontal_  = padded_image_folded_by_horizontal[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
    image_folded_by_vertical_    = padded_image_folded_by_vertical  [ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
    image_folded_by_depth_       = padded_image_folded_by_depth     [ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
    image_folded_by_all1_        = padded_image_folded_by_all1      [ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];

    one_folded_by_asymm_cross_kernel_       = asymm_padded_one_folded_by_asymm_cross_kernel     [ boost::indices[asymm_axis_subrange[0]][asymm_axis_subrange[1]][asymm_axis_subrange[2]] ];
    one_folded_by_asymm_one_kernel_         = asymm_padded_one_folded_by_asymm_one_kernel       [ boost::indices[asymm_axis_subrange[0]][asymm_axis_subrange[1]][asymm_axis_subrange[2]] ];
    one_folded_by_asymm_identity_kernel_    = asymm_padded_one_folded_by_asymm_identity_kernel  [ boost::indices[asymm_axis_subrange[0]][asymm_axis_subrange[1]][asymm_axis_subrange[2]] ];

  }
  
  virtual ~convolutionFixture3D()  { 
    
  };

    
  static const unsigned image_axis_size = ImageDimSize;
  static const unsigned kernel_axis_size = KernelDimSize;

};

typedef convolutionFixture3D<> default_3D_fixture;


}//namespace

#endif











#ifndef _TEST_FIXTURES_H_
#define _TEST_FIXTURES_H_
#include <iostream> 
#include <iomanip> 
#include <vector>
#include <cmath>
//#include "mxn_indexer.hpp"
#include <boost/static_assert.hpp>
#include "boost/multi_array.hpp"

#include <string>
#include "image_stack_utils.h"

// Explanation of the test images:
// image_i.tif		.. the input frame stack
// psf1_i.tif		.. point spread function 1
// psf2_i.tif		.. point spread function 2
// weights_i.tif	.. the weights
// results:
// psi_i.tif		.. the results after the i-th iteration
// Psi_0		.. first guess (all pixels have the same intensity)

namespace multiviewnative {

  template <std::string PathToImages = "" 
	    >
struct deconvolutionFixture
{

  const unsigned  		image_size_				;
  std::vector<int>		image_dims_				;
  image_stack			image_					;
  image_stack			padded_image_				;
  image_stack			image_folded_by_horizontal_		;
  image_stack			image_folded_by_vertical_		;
  image_stack			image_folded_by_depth_			;
  image_stack			image_folded_by_all1_			;

  const unsigned		kernel_size_				;
  std::vector<int>	 	kernel_dims_				;
  image_stack			trivial_kernel_				;
  image_stack			identity_kernel_			;
  image_stack			vertical_kernel_			;
  image_stack			horizont_kernel_			;
  image_stack			depth_kernel_				;
  image_stack			all1_kernel_				;
  
  BOOST_STATIC_ASSERT(KernelDimSize % 2 != 0);

public:
  
  convolutionFixture3D():
    image_size_				((unsigned)std::pow(ImageDimSize,3)),
    image_dims_				(3,ImageDimSize),
    image_				(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    padded_image_			(boost::extents[ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)][ImageDimSize+2*(KernelDimSize/2)]),
    image_folded_by_horizontal_		(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    image_folded_by_vertical_  		(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),       
    image_folded_by_depth_  		(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),       
    image_folded_by_all1_  		(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),       
    kernel_size_			((unsigned)std::pow(KernelDimSize,3)),
    kernel_dims_			(3,KernelDimSize),
    trivial_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    identity_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    vertical_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    horizont_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    depth_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    all1_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize])
  {
    
    //FILL KERNELS
    const unsigned halfKernel  = KernelDimSize/2u;
        
    std::fill(trivial_kernel_.origin()	,trivial_kernel_.origin()	+ kernel_size_	,0.f);
    std::fill(identity_kernel_.origin()	,identity_kernel_.origin()	+ kernel_size_	,0.f);
    std::fill(vertical_kernel_.origin()	,vertical_kernel_.origin()	+ kernel_size_	,0.f);
    std::fill(depth_kernel_.origin()	,depth_kernel_.origin()		+ kernel_size_	,0.f);
    std::fill(horizont_kernel_.origin()	,horizont_kernel_.origin()	+ kernel_size_	,0.f);
    std::fill(all1_kernel_.origin()	,all1_kernel_.origin()		+ kernel_size_	,1.f);

    identity_kernel_.data()[kernel_size_/2]=1.; 

    for(unsigned int index = 0;index<KernelDimSize;++index){
      horizont_kernel_[index][halfKernel][halfKernel] = float(index+1);
      vertical_kernel_[halfKernel][index][halfKernel] = float(index+1);
      depth_kernel_   [halfKernel][halfKernel][index] = float(index+1);
    }
    
    //FILL IMAGES
    unsigned padded_image_axis = ImageDimSize+2*halfKernel;
    unsigned padded_image_size = std::pow(padded_image_axis,3);
    std::fill(image_.origin(),         image_.origin()         +  image_size_,        0.f  );
    std::fill(padded_image_.origin(),  padded_image_.origin()  +  padded_image_size,  0.f  );

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

    //CONVOLVE
    float newValue = 0.;
    float kernel_value  = 0.f;
    float image_value   = 0.f;

    for(int z_index = halfKernel;z_index<int(padded_image_axis-halfKernel);++z_index){
      for(int y_index = halfKernel;y_index<int(padded_image_axis-halfKernel);++y_index){
	for(int x_index = halfKernel;x_index<int(padded_image_axis-halfKernel);++x_index){
	  	  
	  padded_image_folded_by_horizontal[x_index][y_index][z_index] = 0.f;
	  padded_image_folded_by_vertical[x_index][y_index][z_index] = 0.f;
	  padded_image_folded_by_depth[x_index][y_index][z_index] = 0.f;
	  padded_image_folded_by_all1[x_index][y_index][z_index] = 0.f;

	  for(int kindex = 0;kindex<KernelDimSize;++kindex){
	    //convolution in x
	    kernel_value  =  horizont_kernel_[KernelDimSize-1-kindex][halfKernel][halfKernel]	;
	    image_value   =  padded_image_[x_index-halfKernel+kindex][y_index][z_index]		;
	    padded_image_folded_by_horizontal[x_index][y_index][z_index] += kernel_value*image_value;

	    //convolution in y
	    kernel_value  = vertical_kernel_[halfKernel][KernelDimSize-1-kindex][halfKernel];
	    image_value   = padded_image_[x_index][y_index-halfKernel+kindex][z_index];
	    padded_image_folded_by_vertical[x_index][y_index][z_index] += kernel_value*image_value;
	      

	    //convolution in z
	    kernel_value  = depth_kernel_[halfKernel][halfKernel][KernelDimSize-1-kindex];
	    image_value   = padded_image_[x_index][y_index][z_index-halfKernel+kindex];
	    padded_image_folded_by_depth[x_index][y_index][z_index] += kernel_value*image_value;
	      
	  }
  

	  newValue = 0.;
	  for(int z_kernel = -(int)halfKernel;z_kernel<=((int)halfKernel);++z_kernel){
	    for(int y_kernel = -(int)halfKernel;y_kernel<=((int)halfKernel);++y_kernel){
	      for(int x_kernel = -(int)halfKernel;x_kernel<=((int)halfKernel);++x_kernel){
		newValue += padded_image_[x_index+x_kernel][y_index+y_kernel][z_index+z_kernel]*all1_kernel_[halfKernel+x_kernel][halfKernel+y_kernel][halfKernel+z_kernel];
	      }
	    }
	  }
	  padded_image_folded_by_all1[x_index][y_index][z_index] = newValue;
	
	}
     
      }
    }
    
    //EXTRACT NON-PADDED CONTENT FROM CONVOLVED IMAGE STACKS
    image_folded_by_horizontal_  = padded_image_folded_by_horizontal[ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
    image_folded_by_vertical_    = padded_image_folded_by_vertical  [ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
    image_folded_by_depth_       = padded_image_folded_by_depth     [ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];
    image_folded_by_all1_        = padded_image_folded_by_all1      [ boost::indices[axis_subrange][axis_subrange][axis_subrange] ];


  }
  
  virtual ~convolutionFixture3D()  { 
    
  };

  template <typename IntensityT, typename DimT>
  void print_3d_structure(const IntensityT* _image=0, const DimT* _dimensions=0, bool _print_flat = false) const {

    DimT image_index = 0;
    if(!_image){
      std::cerr << "Unable to print 3d structure!\n";
      return;
    }

    for(DimT z_index = 0;z_index<(_dimensions[2]);++z_index){
      std::cout << "z="<< z_index << "\n" << "x" << std::setw(8) << " ";
      for(DimT x_index = 0;x_index<(_dimensions[0]);++x_index){
	std::cout << std::setw(8) << x_index << " ";
      }
      std::cout << "\n\n";
      for(DimT y_index = 0;y_index<(_dimensions[1]);++y_index){
	std::cout << "y["<< std::setw(5) << y_index << "] ";
	for(DimT x_index = 0;x_index<(_dimensions[0]);++x_index){
	  
	  //FIXME: imposes a storage order
	  image_index=x_index;
	  image_index += y_index*_dimensions[0];
	  image_index += z_index*_dimensions[0]*_dimensions[1] ;
	  
	  std::cout << std::setw(8) << _image[image_index] << " ";
	}

	std::cout << "\n";
      }
      std::cout << "\n";
    }

    if(_print_flat){
      DimT image_size = _dimensions[0]*_dimensions[1]*_dimensions[2];
      std::cout << "flat memory storage:\n";
      for(DimT index = 0;index<image_size;++index){
	std::cout << std::setw(6) << _image[index] << " ";
	if((index % 15u == 0) && index>0)
	  std::cout << "\n";
      }
      std::cout << "\n";
    }
  };



  void print_image(const float* _image=0) const {
    
    const float * image_ptr = 0;
    if(!_image)
      image_ptr = &image_.data()[0];
    else
      image_ptr = _image;
    
    print_3d_structure(image_ptr, image_dims_.data());
          
  };
    
  void print_kernel(const float* _kernel=0) const {

    const float * kernel_ptr = 0;
    if(!_kernel)
      kernel_ptr = &identity_kernel_.data()[0];
    else
      kernel_ptr = _kernel;
    
    print_3d_structure(kernel_ptr, kernel_dims_.data());

  };

  
  static const unsigned image_axis_size = ImageDimSize;
  static const unsigned kernel_axis_size = KernelDimSize;

};

typedef convolutionFixture3D<> default_3D_fixture;


}

#endif

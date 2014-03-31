#ifndef _TEST_FIXTURES_H_
#define _TEST_FIXTURES_H_
#include <iostream> 
#include <iomanip> 
#include <vector>
#include <cmath>
//#include "mxn_indexer.hpp"
#include <boost/static_assert.hpp>
#include "boost/multi_array.hpp"

//http://www.boost.org/doc/libs/1_55_0/libs/multi_array/doc/user.html
//http://stackoverflow.com/questions/2168082/how-to-rewrite-array-from-row-order-to-column-order

namespace multiviewnative {

typedef boost::multi_array<float, 3> image_stack;

template <unsigned short KernelDimSize = 3, 
	  unsigned ImageDimSize = 8
	  >
struct convolutionFixture3D
{

  

  const unsigned  		image_size_				;
  std::vector<int>		image_dims_				;
  image_stack			image_					;
  image_stack			padded_image_				;
  image_stack			padded_image_folded_by_horizontal_	;
  image_stack			padded_image_folded_by_vertical_	;
  image_stack			padded_image_folded_by_depth_		;
  image_stack			padded_image_folded_by_all1_		;

  const unsigned		kernel_size_				;
  std::vector<int>	 	kernel_dims_			;
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
    padded_image_			(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    padded_image_folded_by_horizontal_	(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),
    padded_image_folded_by_vertical_  	(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),       
    padded_image_folded_by_depth_  	(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),       
    padded_image_folded_by_all1_  	(boost::extents[ImageDimSize][ImageDimSize][ImageDimSize]),       
    kernel_size_		((unsigned)std::pow(KernelDimSize,3)),
    kernel_dims_		(3,KernelDimSize),
    identity_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    vertical_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    horizont_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    depth_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    all1_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize])
  {
    
    //FILL KERNELS
    const unsigned halfKernel  = KernelDimSize/2u;
        
    std::fill(identity_kernel_.origin()	,identity_kernel_.origin()	+ kernel_size_	,0.f);
    std::fill(vertical_kernel_.origin()	,vertical_kernel_.origin()	+ kernel_size_	,0.f);
    std::fill(depth_kernel_.origin()	,depth_kernel_.origin()		+ kernel_size_	,0.f);
    std::fill(horizont_kernel_.origin()	,horizont_kernel_.origin()	+ kernel_size_	,0.f);
    std::fill(all1_kernel_.origin()	,all1_kernel_.origin()		+ kernel_size_	,1.f);

    identity_kernel_.data()[kernel_size_/2]=1.; 

    for(unsigned int index = 0;index<KernelDimSize;++index){
      horizont_kernel_[index][halfKernel][halfKernel] = float(index+1);
      vertical_kernel_[halfKernel][index][halfKernel] = float(index+1);
      depth_kernel_   [halfKernel][halfKernel][index]	= float(index+1);
    }
    
    //FILL IMAGES
    std::fill(image_.origin(),         image_.origin()         +  image_size_,  42.f  );
    std::fill(padded_image_.origin(),  padded_image_.origin()  +  image_size_,  0.f   );

    unsigned image_index=0;
    for(int z_index = halfKernel;z_index<(image_dims_[2]-halfKernel);++z_index){
      for(int y_index = halfKernel;y_index<(image_dims_[1]-halfKernel);++y_index){
	for(int x_index = halfKernel;x_index<(image_dims_[0]-halfKernel);++x_index){
	  image_index=x_index;
	  image_index += y_index*image_dims_[0];
	  image_index += z_index*image_dims_[0]*image_dims_[1] ;

	  padded_image_[x_index][y_index][z_index] = float(image_index);
	
	}
     
      }
    }

    std::copy(padded_image_.origin(),  padded_image_.origin()  +  image_size_,  padded_image_folded_by_horizontal_.origin());
    std::copy(padded_image_.origin(),  padded_image_.origin()  +  image_size_,  padded_image_folded_by_vertical_.origin());
    std::copy(padded_image_.origin(),  padded_image_.origin()  +  image_size_,  padded_image_folded_by_depth_.origin());
    std::copy(padded_image_.origin(),  padded_image_.origin()  +  image_size_,  padded_image_folded_by_all1_.origin());
    

    //CONVOLVE

    float newValue = 0.;
    float kernel_value  = 0.f;
    float image_value   = 0.f;

    for(int z_index = halfKernel;z_index<(image_dims_[2]-halfKernel);++z_index){
      for(int y_index = halfKernel;y_index<(image_dims_[1]-halfKernel);++y_index){
	for(int x_index = halfKernel;x_index<(image_dims_[0]-halfKernel);++x_index){
	  	  
	  padded_image_folded_by_horizontal_[x_index][y_index][z_index] = 0.f;
	  padded_image_folded_by_vertical_[x_index][y_index][z_index] = 0.f;
	  padded_image_folded_by_depth_[x_index][y_index][z_index] = 0.f;
	  padded_image_folded_by_all1_[x_index][y_index][z_index] = 0.f;

	  for(int kindex = 0;kindex<KernelDimSize;++kindex){
	    //convolution in x
	    kernel_value  =  horizont_kernel_[KernelDimSize-1-kindex][halfKernel][halfKernel]	;
	    image_value   =  padded_image_[x_index-halfKernel+kindex][y_index][z_index]		;
	    padded_image_folded_by_horizontal_[x_index][y_index][z_index] += kernel_value*image_value;

	    //convolution in y
	    kernel_value  = vertical_kernel_[halfKernel][KernelDimSize-1-kindex][halfKernel];
	    image_value   = padded_image_[x_index][y_index-halfKernel+kindex][z_index];
	    padded_image_folded_by_vertical_[x_index][y_index][z_index] += kernel_value*image_value;
	      

	    //convolution in z
	    kernel_value  = depth_kernel_[halfKernel][halfKernel][KernelDimSize-1-kindex];
	    image_value   = padded_image_[x_index][y_index][z_index-halfKernel+kindex];
	    padded_image_folded_by_depth_[x_index][y_index][z_index] += kernel_value*image_value;
	      
	  }
  

	  newValue = 0.;
	  for(int z_kernel = -(int)halfKernel;z_kernel<=((int)halfKernel);++z_kernel){
	    for(int y_kernel = -(int)halfKernel;y_kernel<=((int)halfKernel);++y_kernel){
	      for(int x_kernel = -(int)halfKernel;x_kernel<=((int)halfKernel);++x_kernel){
		newValue += padded_image_[x_index+x_kernel][y_index+y_kernel][z_index+z_kernel]*all1_kernel_[halfKernel+x_kernel][halfKernel+y_kernel][halfKernel+z_kernel];
	      }
	    }
	  }
	  padded_image_folded_by_all1_[x_index][y_index][z_index] = newValue;
	    
	

	
	}
     
      }
    }
    
  }
  
  virtual ~convolutionFixture3D()  { 
    
  };
   
  void print_image(const float* _image=0) const {
    if(!_image)
      _image = &padded_image_.data()[0];
    unsigned image_index = 0;

    for(unsigned z_index = 0;z_index<(image_dims_[2]);++z_index){
      std::cout << "z="<< z_index << "\n" << "x" << std::setw(8) << " ";
      for(unsigned x_index = 0;x_index<(image_dims_[0]);++x_index){
	std::cout << std::setw(8) << x_index << " ";
      }
      std::cout << "\n\n";
      for(unsigned y_index = 0;y_index<(image_dims_[1]);++y_index){
	std::cout << "y["<< std::setw(5) << y_index << "] ";
	for(unsigned x_index = 0;x_index<(image_dims_[0]);++x_index){
	  image_index=x_index;
	  image_index += y_index*image_dims_[0];
	  image_index += z_index*image_dims_[0]*image_dims_[1] ;

	  
	  std::cout << std::setw(8) << _image[image_index] << " ";
	}

	std::cout << "\n";
      }
      std::cout << "\n";
    }

  };
    
  void print_kernel(const float* _kernel=0) const {
    if(!_kernel)
      _kernel = &identity_kernel_.data()[0];
    unsigned kernel_index = 0;

    for(unsigned z_index = 0;z_index<(kernel_dims_[2]);++z_index){
      std::cout << "z="<< z_index << "\n" << "x" << std::setw(8) << " ";
      for(unsigned x_index = 0;x_index<(kernel_dims_[0]);++x_index){
	std::cout << std::setw(8) << x_index << " ";
      }
      std::cout << "\n\n";
      for(unsigned y_index = 0;y_index<(kernel_dims_[1]);++y_index){
	std::cout << "y["<< std::setw(5) << y_index << "] ";
	for(unsigned x_index = 0;x_index<(kernel_dims_[0]);++x_index){
	  kernel_index=x_index;
	  kernel_index += y_index*kernel_dims_[0];
	  kernel_index += z_index*kernel_dims_[0]*kernel_dims_[1] ;

	  
	  std::cout << std::setw(8) << _kernel[kernel_index] << " ";
	}

	std::cout << "\n";
      }
      std::cout << "\n";
    }

  };
};

typedef convolutionFixture3D<> default_3D_fixture;

}

#endif

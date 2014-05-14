#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE INDEPENDENT_CPU_CONVOLVE
#include "boost/test/unit_test.hpp"
#include "test_algorithms.hpp"
#include <iomanip>
#include <iostream>

template <unsigned divisor = 2>
struct divide_by_{

  float operator()(const float& _in){
    return _in/divisor;
  }

};

namespace multiviewnative {

template <unsigned short KernelDimSize = 3, 
	  unsigned ImageDimSize = 8
	  >
struct convolutionFixture3D
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

  image_stack			asymm_cross_kernel_			;
  image_stack			asymm_one_kernel_			;
  image_stack			asymm_identity_kernel_			;
  
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
    all1_kernel_			(boost::extents[KernelDimSize][KernelDimSize][KernelDimSize]),
    asymm_cross_kernel_			(boost::extents[KernelDimSize+1][KernelDimSize][KernelDimSize-1]),
    asymm_one_kernel_			(boost::extents[KernelDimSize+1][KernelDimSize][KernelDimSize-1]),
    asymm_identity_kernel_		(boost::extents[KernelDimSize+1][KernelDimSize][KernelDimSize-1])
  {
    
    //FILL KERNELS
    const unsigned halfKernel  = KernelDimSize/2u;
        
    std::fill(trivial_kernel_.origin()       ,trivial_kernel_.origin()       +  kernel_size_                           ,0.f);
    std::fill(identity_kernel_.origin()      ,identity_kernel_.origin()      +  kernel_size_                           ,0.f);
    std::fill(vertical_kernel_.origin()      ,vertical_kernel_.origin()      +  kernel_size_                           ,0.f);
    std::fill(depth_kernel_.origin()         ,depth_kernel_.origin()         +  kernel_size_                           ,0.f);
    std::fill(horizont_kernel_.origin()      ,horizont_kernel_.origin()      +  kernel_size_                           ,0.f);
    std::fill(all1_kernel_.origin()          ,all1_kernel_.origin()          +  kernel_size_                           ,1.f);
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

  
  
  static const unsigned image_axis_size = ImageDimSize;
  static const unsigned kernel_axis_size = KernelDimSize;

};

typedef convolutionFixture3D<> discrete_3D_fixture;


}


BOOST_FIXTURE_TEST_SUITE( convolution_works, multiviewnative::discrete_3D_fixture )

BOOST_AUTO_TEST_CASE( identity_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, identity_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}

BOOST_AUTO_TEST_CASE( horizont_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, horizont_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_folded_by_horizontal_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}

BOOST_AUTO_TEST_CASE( vertical_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, vertical_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_folded_by_vertical_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}

BOOST_AUTO_TEST_CASE( all1_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, all1_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_folded_by_all1_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( asymm_convolution_works, multiviewnative::discrete_3D_fixture )

BOOST_AUTO_TEST_CASE( asymm_identity_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets(3);

  for(unsigned i = 0;i<3;++i)
      offsets[i] = asymm_one_kernel_.shape()[i]/2;
  

  multiviewnative::convolve(image_, asymm_one_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_NE(sum_expected, sum_received);

}

BOOST_AUTO_TEST_CASE( asymm_one_convolve_replicate )
{

  using namespace multiviewnative;

  multiviewnative::image_stack image = image_;
  std::fill(image.data(), image.data() + image.num_elements(), 0.f);
  image[(image_.shape()[0]-1)/2][(image_.shape()[1]-1)/2][(image_.shape()[2]-1)/2] = 1.f;

  multiviewnative::image_stack result = image;
  
  std::vector<int> offsets(3);

  for(unsigned i = 0;i<3;++i)
      offsets[i] = asymm_one_kernel_.shape()[i]/2;
  

  multiviewnative::convolve(image, asymm_one_kernel_, 
			    result,
			    offsets);

  float sum_expected = std::accumulate(asymm_one_kernel_.data(),asymm_one_kernel_.data() + asymm_one_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(result.data(),result.data() + result.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, 0.0001f);

}

BOOST_AUTO_TEST_CASE( asymm_cross_convolve_replicate )
{

  using namespace multiviewnative;

  multiviewnative::image_stack image = image_;
  std::fill(image.data(), image.data() + image.num_elements(), 0.f);
  image[(image_.shape()[0]-1)/2][(image_.shape()[1]-1)/2][(image_.shape()[2]-1)/2] = 1.f;

  multiviewnative::image_stack result = image;
  
  std::vector<int> offsets(3);

  for(unsigned i = 0;i<3;++i)
      offsets[i] = asymm_cross_kernel_.shape()[i]/2;
  

  multiviewnative::convolve(image, asymm_cross_kernel_, 
			    result,
			    offsets);

  float sum_expected = std::accumulate(asymm_cross_kernel_.data(),asymm_cross_kernel_.data() + asymm_cross_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(result.data(),result.data() + result.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, 0.0001f);

}
BOOST_AUTO_TEST_SUITE_END()

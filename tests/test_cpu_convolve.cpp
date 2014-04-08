#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <cmath>
#include "multiviewnative.h"
#include "fftw3.h"

typedef multiviewnative::image_stack::array_view<3>::type subarray_view;
typedef boost::multi_array_types::index_range range;

using namespace multiviewnative;

template <typename MatrixT>
void print_mismatching_items(MatrixT& _reference, MatrixT& _other){
  for(long x=0;x<_reference.shape()[0];++x)
      for(long y=0;y<_reference.shape()[1];++y)
	for(long z=0;z<_reference.shape()[2];++z){
	  float reference = _reference[x][y][z];
	  float to_compared = _other[x][y][z];
	  if(std::fabs(reference - to_compared)>(1e-3*reference) && (std::fabs(reference) > 1e-4 || std::fabs(to_compared)>1e-4)){
	    std::cout << "["<< x<<"]["<< y<<"]["<< z<<"] mismatch, ref: " << reference << " != to_compare: " << to_compared << "\n";
	  }
	}
}

template <typename MatrixT>
void convolute_3d_out_of_place(MatrixT& _image, MatrixT& _kernel){
  
  if(_image.size()!=_kernel.size())
    {
      std::cerr << "received image and kernel of mismatching size!\n";
      return;
    }

  unsigned M,N,K;
  M = _image.shape()[0];
  N = _image.shape()[1];
  K = _image.shape()[2];
  
  unsigned fft_size = M*N*(K/2+1);

  //setup fourier space arrays
  fftwf_complex* image_fourier = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*fft_size));
  fftwf_complex* kernel_fourier = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*fft_size));
  float scale = 1.0 / (M * N * K);

  //define+run forward plans
  fftwf_plan image_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						    _image.data(), image_fourier,
						    FFTW_ESTIMATE);
  fftwf_execute(image_fwd_plan);

  fftwf_plan kernel_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						     _kernel.data(), kernel_fourier,
						     FFTW_ESTIMATE);
  fftwf_execute(kernel_fwd_plan);


  //multiply
  for(unsigned index = 0;index < fft_size;++index){
    float real = image_fourier[index][0]*kernel_fourier[index][0] - image_fourier[index][1]*kernel_fourier[index][1];
    float imag = image_fourier[index][0]*kernel_fourier[index][1] + image_fourier[index][1]*kernel_fourier[index][0];
    image_fourier[index][0] = real;
    image_fourier[index][1] = imag;
  }
  
  fftwf_destroy_plan(kernel_fwd_plan);
  fftwf_destroy_plan(image_fwd_plan);
  
  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(M, N, K,
						    image_fourier, _image.data(),
						    FFTW_ESTIMATE);
  fftwf_execute(image_rev_plan);
  
  for(unsigned index = 0;index < _image.num_elements();++index){
    _image.data()[index]*=scale;
  }
  
  fftwf_destroy_plan(image_rev_plan);
  fftwf_free(image_fourier);
  fftwf_free(kernel_fourier);
}





template <typename MatrixT>
void convolute_3d_in_place(MatrixT& _image, const MatrixT& _kernel){
  
  if(_image.size()==_kernel.size())
    {
      std::cerr << "received image and kernel of matching size, this makes preparing the kernel impossible!\nExiting.\n";
      return;
    }

  if(MatrixT::dimensionality!=3)
    {
      std::cerr << "received image and kernel of dimension "<< MatrixT::dimensionality<<" that cannot be processed by convolute_3d_in_place!\n";
      return;
    }
  
  std::vector<unsigned> origin_image_extents(MatrixT::dimensionality);
  std::copy(_image.shape(), _image.shape() + MatrixT::dimensionality, origin_image_extents.begin());

  std::vector<unsigned> origin_kernel_extents(MatrixT::dimensionality);
  std::copy(_kernel.shape(), _kernel.shape() + MatrixT::dimensionality, origin_kernel_extents.begin());

  
  ///////////////////////////////////////////////////////////////////////////
  //CALCULATE PADDING EXTENT
  std::vector<unsigned> common_extents(MatrixT::dimensionality);
  std::transform(origin_image_extents.begin(), origin_image_extents.end(), origin_kernel_extents.begin(), 
		 common_extents.begin(), add_minus_1<unsigned>());
  
  std::vector<unsigned> common_offsets(MatrixT::dimensionality);
  std::transform(origin_kernel_extents.begin(), origin_kernel_extents.end(), common_offsets.begin(), minus_1_div_2<unsigned>());
  
  ///////////////////////////////////////////////////////////////////////////
  //PADD IMAGE
  image_stack padded_image(common_extents, _image.storage_order());
  image_stack_view subview_padded_image =  padded_image[ boost::indices[range(common_offsets[0], common_offsets[0]+origin_image_extents[0])][range(common_offsets[1], common_offsets[1]+origin_image_extents[1])][range(common_offsets[2], common_offsets[2]+origin_image_extents[2])] ];
  subview_padded_image = _image;
  unsigned long size_of_transform = padded_image.num_elements();

  ///////////////////////////////////////////////////////////////////////////
  //PADD KERNEL
  image_stack padded_kernel(common_extents, _kernel.storage_order());
  for(long z=0;z<origin_kernel_extents[2];++z)
    for(long y=0;y<origin_kernel_extents[1];++y)
      for(long x=0;x<origin_kernel_extents[0];++x){
	long intermediate_x = x - origin_kernel_extents[0]/2L;
	long intermediate_y = y - origin_kernel_extents[1]/2L;
	long intermediate_z = z - origin_kernel_extents[2]/2L;

	intermediate_x =(intermediate_x<0) ? intermediate_x + common_extents[0]: intermediate_x;
	intermediate_y =(intermediate_y<0) ? intermediate_y + common_extents[1]: intermediate_y;
	intermediate_z =(intermediate_z<0) ? intermediate_z + common_extents[2]: intermediate_z;
	
	padded_kernel[intermediate_x][intermediate_y][intermediate_z] = _kernel[x][y][z];
      }


  ///////////////////////////////////////////////////////////////////////////
  //RESIZE ALL TO ALLOW FFTW INPLACE TRANSFORM
  std::vector<unsigned> inplace_extents = adapt_extents_for_fftw_inplace(_image.storage_order(),common_extents);
  padded_image.resize(boost::extents[inplace_extents[0]][inplace_extents[1]][inplace_extents[2]]);
  padded_kernel.resize(boost::extents[inplace_extents[0]][inplace_extents[1]][inplace_extents[2]]);
  

  float scale = 1.0 / (size_of_transform);
  fftwf_complex* complex_image_fourier  = (fftwf_complex*)padded_image.data();
  fftwf_complex* complex_kernel_fourier = (fftwf_complex*)padded_kernel.data();

  //define+run forward plans
  fftwf_plan image_fwd_plan = fftwf_plan_dft_r2c_3d(common_extents[0], common_extents[1], common_extents[2],
						    padded_image.data(), complex_image_fourier,
						    FFTW_ESTIMATE);
  fftwf_execute(image_fwd_plan);

  fftwf_plan kernel_fwd_plan = fftwf_plan_dft_r2c_3d(common_extents[0], common_extents[1], common_extents[2],
						     padded_kernel.data(), complex_kernel_fourier,
						     FFTW_ESTIMATE);
  fftwf_execute(kernel_fwd_plan);


  //multiply
 
  unsigned fourier_num_elements = padded_image.num_elements()/2;
  for(unsigned index = 0;index < fourier_num_elements;++index){
    float real = complex_image_fourier[index][0]*complex_kernel_fourier[index][0] - complex_image_fourier[index][1]*complex_kernel_fourier[index][1];
    float imag = complex_image_fourier[index][0]*complex_kernel_fourier[index][1] + complex_image_fourier[index][1]*complex_kernel_fourier[index][0];
    complex_image_fourier[index][0] = real;
    complex_image_fourier[index][1] = imag;
  }
  
  fftwf_destroy_plan(kernel_fwd_plan);
  fftwf_destroy_plan(image_fwd_plan);
    
  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(common_extents[0], common_extents[1], common_extents[2],
						     complex_image_fourier, padded_image.data(),
						    FFTW_ESTIMATE);
  fftwf_execute(image_rev_plan);
  
  for(unsigned index = 0;index < padded_image.num_elements();++index){
    padded_image.data()[index]*=scale;
  }

  fftwf_destroy_plan(image_rev_plan);

  _image = padded_image[ boost::indices[range(common_offsets[0], common_offsets[0]+origin_image_extents[0])][range(common_offsets[1], common_offsets[1]+origin_image_extents[1])][range(common_offsets[2], common_offsets[2]+origin_image_extents[2])] ];

}



BOOST_FIXTURE_TEST_SUITE( cpu_out_of_place_convolution, multiviewnative::default_3D_fixture )
BOOST_AUTO_TEST_CASE( fft_ifft_image )
{

  unsigned M = multiviewnative::default_3D_fixture::image_axis_size, N = multiviewnative::default_3D_fixture::image_axis_size, K = multiviewnative::default_3D_fixture::image_axis_size;
  
  unsigned fft_out_of_place_size = M*N*(K/2+1);
  fftwf_complex* image_fourier = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*fft_out_of_place_size);
  float scale = 1.0 / (image_size_);
  
  fftwf_plan image_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						    image_.data(), image_fourier,
						    FFTW_ESTIMATE);
  fftwf_execute(image_fwd_plan);
  fftwf_destroy_plan(image_fwd_plan);
  
  float* image_result = (float *) fftwf_malloc(sizeof(float)*image_size_);
  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(M, N, K,
						    image_fourier, image_result,
						    FFTW_ESTIMATE);
  fftwf_execute(image_rev_plan);
  
  for(unsigned index = 0;index < image_size_;++index){
    image_result[index]*=scale;
  }
  
  float sum_result   = std::accumulate(image_result, image_result + image_size_, 0.f);
  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_size_, 0.f);
  
  BOOST_CHECK_CLOSE(sum_result, sum_original, 0.000001);
  
  fftwf_destroy_plan(image_rev_plan);
  fftwf_free(image_result);
  fftwf_free(image_fourier);


}


BOOST_AUTO_TEST_CASE( trivial_convolve )
{
  

  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data
  unsigned common_size = image_axis_size + kernel_axis_size - 1;
  unsigned image_offset = (common_size - image_axis_size)/2;

  multiviewnative::image_stack  padded_image(   boost::extents[common_size][common_size][common_size]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[common_size][common_size][common_size]);
  
  //padd image by zero
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ]];
  padded_image_view = image_;

  //padd and shift the kernel
  //not required here
  std::fill(padded_kernel.origin(), padded_kernel.origin()+padded_kernel.size(),0.f);
  ///////////////////////////////////////////////////////////////////////////
  //based upon from 
  //http://www.fftw.org/fftw2_doc/fftw_2.html
  unsigned M = multiviewnative::default_3D_fixture::image_axis_size, N = multiviewnative::default_3D_fixture::image_axis_size, K = multiviewnative::default_3D_fixture::image_axis_size;

  //see http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data
  //For out-of-place transforms, this is the end of the story: the real data is stored as a row-major array of size n0 × n1 × n2 × … × nd-1 and the complex data is stored as a row-major array of size n0 × n1 × n2 × … × (nd-1/2 + 1).
  // For in-place transforms, however, extra padding of the real-data array is necessary because the complex array is larger than the real array, and the two arrays share the same memory locations. Thus, for in-place transforms, the final dimension of the real-data array must be padded with extra values to accommodate the size of the complex data—two values if the last dimension is even and one if it is odd. That is, the last dimension of the real data must physically contain 2 * (nd-1/2+1)double values (exactly enough to hold the complex data). This physical array size does not, however, change the logical array size—only nd-1values are actually stored in the last dimension, and nd-1is the last dimension passed to the plan-creation routine.
  unsigned fft_size = M*N*(K/2+1);
  fftwf_complex* image_fourier = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*fft_size));
  fftwf_complex* kernel_fourier = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*fft_size));
  float scale = 1.0 / (M * N * K);

  
  fftwf_plan image_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						    padded_image.data(), image_fourier,
						    FFTW_ESTIMATE);
  fftwf_execute(image_fwd_plan);

  fftwf_plan kernel_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						     padded_kernel.data(), kernel_fourier,
						     FFTW_ESTIMATE);
  fftwf_execute(kernel_fwd_plan);


  
  
  for(unsigned index = 0;index < fft_size;++index){
    float real = image_fourier[index][0]*kernel_fourier[index][0] - image_fourier[index][1]*kernel_fourier[index][1];
    float imag = image_fourier[index][0]*kernel_fourier[index][1] + image_fourier[index][1]*kernel_fourier[index][0];
    image_fourier[index][0] = real;
    image_fourier[index][1] = imag;
  }
  
  fftwf_destroy_plan(kernel_fwd_plan);
  fftwf_destroy_plan(image_fwd_plan);
  
  float* image_result = (float *) fftwf_malloc(sizeof(float)*image_size_);
  
  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(M, N, K,
						    image_fourier, image_result,
						    FFTW_ESTIMATE);
  fftwf_execute(image_rev_plan);
  
  float sum = std::accumulate(image_result, image_result + image_size_,0.f)*scale;
  BOOST_CHECK_CLOSE(sum, 0.f, .00001);

  fftwf_destroy_plan(image_rev_plan);
  fftwf_free(image_result);
  fftwf_free(image_fourier);
  fftwf_free(kernel_fourier);

}



BOOST_AUTO_TEST_CASE( convolve_by_identity )
{
  
  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data
  unsigned common_axis_size = image_axis_size + kernel_axis_size - 1;
  unsigned common_size = common_axis_size*common_axis_size*common_axis_size;
  unsigned image_offset = (common_axis_size - image_axis_size)/2;

  multiviewnative::image_stack  padded_image(   boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  
  //insert input into zero-padded stack
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ]];
  padded_image_view = image_;

  //insert kernel into zero-padded stack
  std::fill(padded_kernel.origin(), padded_kernel.origin()+common_size,0.f);

  //shift the kernel cyclic
  for(long x=0;x<kernel_axis_size;++x)
    for(long y=0;y<kernel_axis_size;++y)
      for(long z=0;z<kernel_axis_size;++z){
	long intermediate_x = x - kernel_axis_size/2L;
	long intermediate_y = y - kernel_axis_size/2L;
	long intermediate_z = z - kernel_axis_size/2L;
	
	intermediate_x =(intermediate_x<0) ? intermediate_x + common_axis_size: intermediate_x;
	intermediate_y =(intermediate_y<0) ? intermediate_y + common_axis_size: intermediate_y;
	intermediate_z =(intermediate_z<0) ? intermediate_z + common_axis_size: intermediate_z;

	padded_kernel[intermediate_x][intermediate_y][intermediate_z] = identity_kernel_[x][y][z];
      }
  
  unsigned M = common_axis_size, N = common_axis_size, K = common_axis_size;
  unsigned fft_size = M*N*(K/2+1);

  //setup fourier space arrays
  fftwf_complex* image_fourier = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*fft_size));
  fftwf_complex* kernel_fourier = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex)*fft_size));
  float scale = 1.0 / (M * N * K);

  //define+run forward plans
  fftwf_plan image_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						    padded_image.data(), image_fourier,
						    FFTW_ESTIMATE);
  fftwf_execute(image_fwd_plan);

  fftwf_plan kernel_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						     padded_kernel.data(), kernel_fourier,
						     FFTW_ESTIMATE);
  fftwf_execute(kernel_fwd_plan);


  //multiply
  for(unsigned index = 0;index < fft_size;++index){
    float real = image_fourier[index][0]*kernel_fourier[index][0] - image_fourier[index][1]*kernel_fourier[index][1];
    float imag = image_fourier[index][0]*kernel_fourier[index][1] + image_fourier[index][1]*kernel_fourier[index][0];
    image_fourier[index][0] = real;
    image_fourier[index][1] = imag;
  }
  
  fftwf_destroy_plan(kernel_fwd_plan);
  fftwf_destroy_plan(image_fwd_plan);
    
  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(M, N, K,
						    image_fourier, padded_image.data(),
						    FFTW_ESTIMATE);
  fftwf_execute(image_rev_plan);
  
  for(unsigned index = 0;index < common_size;++index){
    padded_image.data()[index]*=scale;
  }
  
  range image_segment_range = range(image_offset,image_offset+image_axis_size);
  subarray_view  result_image_view   =  padded_image[ boost::indices[  image_segment_range     ][  image_segment_range     ][ image_segment_range    ] ];

  image_stack result_image = result_image_view;
  
  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);
  float sum = std::accumulate(result_image.origin(), result_image.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);

  fftwf_destroy_plan(image_rev_plan);
  fftwf_free(image_fourier);
  fftwf_free(kernel_fourier);

}

BOOST_AUTO_TEST_CASE( convolve_by_identity_from_external )
{
  
  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data
  unsigned common_axis_size = image_axis_size + kernel_axis_size - 1;
  unsigned common_size = common_axis_size*common_axis_size*common_axis_size;
  unsigned image_offset = (common_axis_size - image_axis_size)/2;

  multiviewnative::image_stack  padded_image(   boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  
  //insert input into zero-padded stack
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ]];
  padded_image_view = image_;

  //insert kernel into zero-padded stack
  std::fill(padded_kernel.origin(), padded_kernel.origin()+common_size,0.f);

  //shift the kernel cyclic
  for(long x=0;x<kernel_axis_size;++x)
    for(long y=0;y<kernel_axis_size;++y)
      for(long z=0;z<kernel_axis_size;++z){
	long intermediate_x = x - kernel_axis_size/2L;
	long intermediate_y = y - kernel_axis_size/2L;
	long intermediate_z = z - kernel_axis_size/2L;
	
	intermediate_x =(intermediate_x<0) ? intermediate_x + common_axis_size: intermediate_x;
	intermediate_y =(intermediate_y<0) ? intermediate_y + common_axis_size: intermediate_y;
	intermediate_z =(intermediate_z<0) ? intermediate_z + common_axis_size: intermediate_z;

	padded_kernel[intermediate_x][intermediate_y][intermediate_z] = identity_kernel_[x][y][z];
      }
  

  convolute_3d_out_of_place(padded_image,padded_kernel);
  
  range image_segment_range = range(image_offset,image_offset+image_axis_size);
  subarray_view  result_image_view   =  padded_image[ boost::indices[  image_segment_range     ][  image_segment_range     ][ image_segment_range    ] ];

  image_stack result_image = result_image_view;
  
  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);
  float sum = std::accumulate(result_image.origin(), result_image.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}



BOOST_AUTO_TEST_CASE( convolve_by_horizontal )
{
  
  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data
  unsigned common_axis_size = image_axis_size + kernel_axis_size - 1;
  unsigned common_size = common_axis_size*common_axis_size*common_axis_size;
  unsigned image_offset = (common_axis_size - image_axis_size)/2;

  multiviewnative::image_stack  padded_image(   boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  
  //insert input into zero-padded stack
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ]];
  padded_image_view = image_;

  //insert kernel into zero-padded stack
  std::fill(padded_kernel.origin(), padded_kernel.origin()+common_size,0.f);

  //shift the kernel cyclic
  for(long x=0;x<kernel_axis_size;++x)
    for(long y=0;y<kernel_axis_size;++y)
      for(long z=0;z<kernel_axis_size;++z){
	long intermediate_x = x - kernel_axis_size/2L;
	long intermediate_y = y - kernel_axis_size/2L;
	long intermediate_z = z - kernel_axis_size/2L;
	
	intermediate_x =(intermediate_x<0) ? intermediate_x + common_axis_size: intermediate_x;
	intermediate_y =(intermediate_y<0) ? intermediate_y + common_axis_size: intermediate_y;
	intermediate_z =(intermediate_z<0) ? intermediate_z + common_axis_size: intermediate_z;

	padded_kernel[intermediate_x][intermediate_y][intermediate_z] = horizont_kernel_[x][y][z];
      }
  
   convolute_3d_out_of_place(padded_image,padded_kernel);

  // multiviewnative::image_stack_ref result(image_result,boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  range image_segment_range = range(image_offset,image_offset+image_axis_size);
  subarray_view  result_image_view   =  padded_image[ boost::indices[  image_segment_range     ][  image_segment_range     ][ image_segment_range    ] ];

  //copy awkward here
  image_stack result_image = result_image_view;
  

  float sum_original = std::accumulate(image_folded_by_horizontal_.origin(), image_folded_by_horizontal_.origin() + image_size_,0.f);
  float sum = std::accumulate(result_image.origin(), result_image.origin() + image_size_,0.f);

  

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}


BOOST_AUTO_TEST_CASE( convolve_by_vertical )
{
  
  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data
  unsigned common_axis_size = image_axis_size + kernel_axis_size - 1;
  unsigned common_size = common_axis_size*common_axis_size*common_axis_size;
  unsigned image_offset = (common_axis_size - image_axis_size)/2;

  multiviewnative::image_stack  padded_image(   boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  
  //insert input into zero-padded stack
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ]];
  padded_image_view = image_;

  //insert kernel into zero-padded stack
  std::fill(padded_kernel.origin(), padded_kernel.origin()+common_size,0.f);

  //shift the kernel cyclic
  for(long x=0;x<kernel_axis_size;++x)
    for(long y=0;y<kernel_axis_size;++y)
      for(long z=0;z<kernel_axis_size;++z){
	long intermediate_x = x - kernel_axis_size/2L;
	long intermediate_y = y - kernel_axis_size/2L;
	long intermediate_z = z - kernel_axis_size/2L;
	
	intermediate_x =(intermediate_x<0) ? intermediate_x + common_axis_size: intermediate_x;
	intermediate_y =(intermediate_y<0) ? intermediate_y + common_axis_size: intermediate_y;
	intermediate_z =(intermediate_z<0) ? intermediate_z + common_axis_size: intermediate_z;

	padded_kernel[intermediate_x][intermediate_y][intermediate_z] = vertical_kernel_[x][y][z];
      }
  
  convolute_3d_out_of_place(padded_image,padded_kernel);

  // multiviewnative::image_stack_ref result(image_result,boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  range image_segment_range = range(image_offset,image_offset+image_axis_size);
  subarray_view  result_image_view   =  padded_image[ boost::indices[  image_segment_range     ][  image_segment_range     ][ image_segment_range    ] ];

  //copy awkward here
  image_stack result_image = result_image_view;

  float sum_original = std::accumulate(image_folded_by_vertical_.origin(), image_folded_by_vertical_.origin() + image_size_,0.f);
  float sum = std::accumulate(result_image.origin(), result_image.origin() + image_size_,0.f);

  

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}


BOOST_AUTO_TEST_CASE( convolve_by_all1 )
{
  
  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data
  unsigned common_axis_size = image_axis_size + kernel_axis_size - 1;
  unsigned common_size = common_axis_size*common_axis_size*common_axis_size;
  unsigned image_offset = (common_axis_size - image_axis_size)/2;

  multiviewnative::image_stack  padded_image(   boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  
  //insert input into zero-padded stack
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ]];
  padded_image_view = image_;

  //insert kernel into zero-padded stack
  std::fill(padded_kernel.origin(), padded_kernel.origin()+common_size,0.f);

  //shift the kernel cyclic
  for(long x=0;x<kernel_axis_size;++x)
    for(long y=0;y<kernel_axis_size;++y)
      for(long z=0;z<kernel_axis_size;++z){
	long intermediate_x = x - kernel_axis_size/2L;
	long intermediate_y = y - kernel_axis_size/2L;
	long intermediate_z = z - kernel_axis_size/2L;
	
	intermediate_x =(intermediate_x<0) ? intermediate_x + common_axis_size: intermediate_x;
	intermediate_y =(intermediate_y<0) ? intermediate_y + common_axis_size: intermediate_y;
	intermediate_z =(intermediate_z<0) ? intermediate_z + common_axis_size: intermediate_z;

	padded_kernel[intermediate_x][intermediate_y][intermediate_z] = all1_kernel_[x][y][z];
      }
  
  convolute_3d_out_of_place(padded_image,padded_kernel);

  // multiviewnative::image_stack_ref result(image_result,boost::extents[common_axis_size][common_axis_size][common_axis_size]);
  range image_segment_range = range(image_offset,image_offset+image_axis_size);
  subarray_view  result_image_view   =  padded_image[ boost::indices[  image_segment_range     ][  image_segment_range     ][ image_segment_range    ] ];

  //copy awkward here
  image_stack result_image = result_image_view;

  float sum_original = std::accumulate(image_folded_by_all1_.origin(), image_folded_by_all1_.origin() + image_size_,0.f);
  float sum = std::accumulate(result_image.origin(), result_image.origin() + image_size_,0.f);
 

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( cpu_in_place_convolution, multiviewnative::default_3D_fixture )
BOOST_AUTO_TEST_CASE( explicit_fft_ifft_image )
{

  unsigned M = multiviewnative::default_3D_fixture::image_axis_size, N = multiviewnative::default_3D_fixture::image_axis_size, K = multiviewnative::default_3D_fixture::image_axis_size;
  
  image_stack padded_image(boost::extents[M][N][2*(K/2 + 1)]);
  image_stack_view padded_image_view = padded_image[ boost::indices[range()][range()][range(0,K)] ];
  padded_image_view = image_;

  float scale = 1.0 / (image_size_);
  
  fftwf_plan image_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						    padded_image.data(), (fftwf_complex*)padded_image.data(),
						    FFTW_ESTIMATE);
  fftwf_execute(image_fwd_plan);
  fftwf_destroy_plan(image_fwd_plan);
  

  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(M, N, K,
						    (fftwf_complex*)padded_image.data(), padded_image.data(),
						    FFTW_ESTIMATE);
  fftwf_execute(image_rev_plan);
  
  for(unsigned index = 0;index < padded_image.num_elements();++index){
    padded_image.data()[index]*=scale;
  }

  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_size_, 0.f);
  
  image_ = padded_image_view;
  float sum_result   = std::accumulate(image_.origin(), image_.origin() + image_size_, 0.f);

  
  BOOST_CHECK_CLOSE(sum_result, sum_original, 0.000001);
  
  fftwf_destroy_plan(image_rev_plan);




}

BOOST_AUTO_TEST_CASE( explicit_convolve_by_identity )
{
  
  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data due to fftw memory restrictions on inplace transforms
  //http://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html#Real_002ddata-DFT-Array-Format
  unsigned last_dim_size_for_inplace = 2*(image_axis_size/2 + 1);

  multiviewnative::image_stack  padded_image(   boost::extents[image_axis_size][image_axis_size][last_dim_size_for_inplace]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[image_axis_size][image_axis_size][last_dim_size_for_inplace]);
  
  //insert input into zero-padded stack
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range()     ][  range()     ][  range(0,image_axis_size)     ]];
  padded_image_view = image_;

  //insert kernel into zero-padded stack
  std::fill(padded_kernel.origin(), padded_kernel.origin()+padded_kernel.num_elements(),0.f);

  //shift the kernel cyclic
  for(long x=0;x<kernel_axis_size;++x)
    for(long y=0;y<kernel_axis_size;++y)
      for(long z=0;z<kernel_axis_size;++z){
	long intermediate_x = x - kernel_axis_size/2L;
	long intermediate_y = y - kernel_axis_size/2L;
	long intermediate_z = z - kernel_axis_size/2L;
	
	intermediate_x =(intermediate_x<0) ? intermediate_x + image_axis_size: intermediate_x;
	intermediate_y =(intermediate_y<0) ? intermediate_y + image_axis_size: intermediate_y;
	intermediate_z =(intermediate_z<0) ? intermediate_z + image_axis_size: intermediate_z;

	padded_kernel[intermediate_x][intermediate_y][intermediate_z] = identity_kernel_[x][y][z];
      }
  
  unsigned M = image_axis_size, N = image_axis_size, K = image_axis_size;
  
  //setup fourier space arrays
  float scale = 1.0 / (M * N * K);

  //define+run forward plans
  fftwf_plan image_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						    padded_image.data(), (fftwf_complex*)padded_image.data(),
						    FFTW_ESTIMATE);
  fftwf_execute(image_fwd_plan);

  fftwf_plan kernel_fwd_plan = fftwf_plan_dft_r2c_3d(M, N, K,
						     padded_kernel.data(), (fftwf_complex*)padded_kernel.data(),
						     FFTW_ESTIMATE);
  fftwf_execute(kernel_fwd_plan);


  //multiply
  fftwf_complex* image_fourier = (fftwf_complex*)padded_image.data();
  fftwf_complex* kernel_fourier = (fftwf_complex*)padded_kernel.data();
  unsigned fourier_num_elements = padded_image.num_elements()/2;
  for(unsigned index = 0;index < fourier_num_elements;++index){
    float real = image_fourier[index][0]*kernel_fourier[index][0] - image_fourier[index][1]*kernel_fourier[index][1];
    float imag = image_fourier[index][0]*kernel_fourier[index][1] + image_fourier[index][1]*kernel_fourier[index][0];
    image_fourier[index][0] = real;
    image_fourier[index][1] = imag;
  }
  
  fftwf_destroy_plan(kernel_fwd_plan);
  fftwf_destroy_plan(image_fwd_plan);
    
  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(M, N, K,
						    (fftwf_complex*)padded_image.data(), padded_image.data(),
						    FFTW_ESTIMATE);
  fftwf_execute(image_rev_plan);
  
  for(unsigned index = 0;index < padded_image.num_elements();++index){
    padded_image.data()[index]*=scale;
  }
  
  
  
  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);
  image_ = padded_image_view ;
  float sum = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);

  fftwf_destroy_plan(image_rev_plan);

}

BOOST_AUTO_TEST_CASE( convolve_by_identity )
{
  
  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  convolute_3d_in_place(image_, identity_kernel_);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);

}

BOOST_AUTO_TEST_CASE( convolve_by_horizontal )
{
  
  
  float sum_original = std::accumulate(image_folded_by_horizontal_.origin(), image_folded_by_horizontal_.origin() + image_size_,0.f);

  convolute_3d_in_place(image_, horizont_kernel_);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);
  

}

BOOST_AUTO_TEST_CASE( convolve_by_vertical )
{
  
  
  float sum_original = std::accumulate(image_folded_by_vertical_.origin(), image_folded_by_vertical_.origin() + image_size_,0.f);

  convolute_3d_in_place(image_, vertical_kernel_);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);
  

}

BOOST_AUTO_TEST_CASE( convolve_by_all1 )
{
  
  float sum_original = std::accumulate(image_folded_by_all1_.origin(), image_folded_by_all1_.origin() + image_size_,0.f);

  
  convolute_3d_in_place(image_, all1_kernel_);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);
  

}


BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( convolution_from_api, multiviewnative::default_3D_fixture )
BOOST_AUTO_TEST_CASE( identity_kernel )
{

  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  cpu_convolve convolver(image_.data(), image_.shape(), identity_kernel_.data(), identity_kernel_.shape());
  convolver.perform();

  float sum = std::accumulate(image_.origin(), image_.origin() + image_size_,0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}
BOOST_AUTO_TEST_SUITE_END()

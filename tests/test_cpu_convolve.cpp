#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "multiviewnative.h"
#include "fftw3.h"

BOOST_FIXTURE_TEST_SUITE( cpu_out_of_place_convolution, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( trivial_convolve )
{
  typedef multiviewnative::image_stack::array_view<3>::type subarray_view;
  typedef boost::multi_array_types::index_range range;

  ///////////////////////////////////////////////////////////////////////////
  //prepare/padd data
  unsigned common_size = image_axis_size + kernel_axis_size - 1;
  unsigned image_offset = (common_size - image_axis_size)/2;

  multiviewnative::image_stack  padded_image(   boost::extents[common_size][common_size][common_size]);
  multiviewnative::image_stack  padded_kernel(  boost::extents[common_size][common_size][common_size]);

  //padd image by zero
  subarray_view  padded_image_view   =  padded_image[   boost::indices[  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ][  range(image_offset,image_offset+image_axis_size)     ]];
  padded_image_view = padded_image_;

  //padd and shift the kernel
  //not required here
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
  
  fftwf_plan image_rev_plan = fftwf_plan_dft_c2r_3d(M, N, K/2+1,
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
  BOOST_FAIL("convolution by unity not implemented yet");
}


BOOST_AUTO_TEST_SUITE_END()

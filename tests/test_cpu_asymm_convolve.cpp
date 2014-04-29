#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"

BOOST_FIXTURE_TEST_SUITE( convolution_works, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( trivial_convolve )
{
  
  float* image = image_.data();
  float* kernel = new float[kernel_size_];
  std::fill(kernel, kernel+kernel_size_,0.f);

  inplace_cpu_convolution(image, &image_dims_[0], 
			  kernel,&kernel_dims_[0],
			  1);

  float sum = std::accumulate(image, image + image_size_,0.f);
  BOOST_CHECK_CLOSE(sum, 0.f, .00001);

  delete [] kernel;
}


BOOST_AUTO_TEST_SUITE_END()

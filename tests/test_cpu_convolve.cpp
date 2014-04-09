#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
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
			  0);

  float sum = std::accumulate(image, image + image_size_,0.f);
  BOOST_CHECK_CLOSE(sum, 0.f, .00001);

  delete [] kernel;
}

BOOST_AUTO_TEST_CASE( identity_convolve )
{
  using namespace multiviewnative;

  float sum_original = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  identity_kernel_.data(),&kernel_dims_[0],
  			  0);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}

BOOST_AUTO_TEST_CASE( horizontal_convolve )
{
  using namespace multiviewnative;

  float sum_original = std::accumulate(image_folded_by_horizontal_.origin(), image_folded_by_horizontal_.origin() + image_folded_by_horizontal_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  horizont_kernel_.data(),&kernel_dims_[0],
  			  0);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}

BOOST_AUTO_TEST_CASE( vertical_convolve )
{
  using namespace multiviewnative;

  float sum_original = std::accumulate(image_folded_by_vertical_.origin(), image_folded_by_vertical_.origin() + image_folded_by_vertical_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  vertical_kernel_.data(),&kernel_dims_[0],
  			  0);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}

BOOST_AUTO_TEST_CASE( all1_convolve )
{
  using namespace multiviewnative;

  float sum_original = std::accumulate(image_folded_by_all1_.origin(), image_folded_by_all1_.origin() + image_folded_by_all1_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  all1_kernel_.data(),&kernel_dims_[0],
  			  0);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}
BOOST_AUTO_TEST_SUITE_END()

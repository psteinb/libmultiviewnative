#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <iostream>
#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
#include "test_algorithms.hpp"
#include "image_stack_utils.h"

typedef multiviewnative::zero_padd<multiviewnative::image_stack> zero_padding;
static multiviewnative::storage local_order = boost::c_storage_order();

using namespace multiviewnative;

BOOST_FIXTURE_TEST_SUITE( convolution_works, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( trivial_convolve )
{

  std::vector<float> kernel(kernel_size_,0.f);
  image_stack input = padded_image_;



  inplace_cpu_convolution(padded_image_.data(), &padded_image_dims_[0], 
			  &kernel[0],&kernel_dims_[0],
			  1);

  float sum = std::accumulate(padded_image_.data(), padded_image_.data() + padded_image_.num_elements(),0.f);
  float alt_sum = 0;
  for( unsigned i = 0;i<padded_image_.num_elements();++i)
    alt_sum += padded_image_.data()[i];

  BOOST_CHECK_CLOSE(alt_sum, 0.f, .001);
  

  try{
    BOOST_REQUIRE_CLOSE(sum, 0.f, .00001);
  }
  catch(...){
    image_stack expected = input;
    std::fill(expected.data(), expected.data() + expected.num_elements(),0);
    std::cout << "input:\n" << input << "\n\n"
	      << "received:" << padded_image_ << "\n\n"
    	      << "expected:" << expected << "\n\n";
  }


}

BOOST_AUTO_TEST_CASE( identity_convolve )
{
  
  image_stack expected = padded_image_;
  float sum_original = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);

  inplace_cpu_convolution(padded_image_.data(), &padded_image_dims_[0], 
  			  identity_kernel_.data(),&kernel_dims_[0],
  			  1);

  image_ = padded_image_[ boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]] ];
  float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);

  try{
    BOOST_REQUIRE_CLOSE(sum, sum_original, .0001);
  }
  catch(...){

    std::cout << "received:" << padded_image_ << "\n\n"
    	      << "expected:" << expected << "\n\n";
  }


}

BOOST_AUTO_TEST_CASE( horizontal_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_horizontal_.origin(), image_folded_by_horizontal_.origin() + image_folded_by_horizontal_.num_elements(),0.f);
  inplace_cpu_convolution(padded_image_.data(), &padded_image_dims_[0], 
  			  horizont_kernel_.data(),&kernel_dims_[0],
  			  1);

  mvn::range axis_subrange = mvn::range(halfKernel,halfKernel+imageDimSize);
  image_ = padded_image_[ boost::indices[axis_subrange][axis_subrange][axis_subrange]];
  float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);

  try{
    BOOST_REQUIRE_CLOSE(sum, sum_original, .00001f);
  }
  catch(...){
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
	      << "\n input:\n" << padded_image_ << "\n"
	      << "\n kernel:\n" << horizont_kernel_
	      << "\n expected:\n" << image_folded_by_horizontal_ << "\n"
	      << "\n received:\n" << image_ << "\n"
      ;
  }
}

BOOST_AUTO_TEST_CASE( vertical_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_vertical_.origin(), image_folded_by_vertical_.origin() + image_folded_by_vertical_.num_elements(),0.f);
  inplace_cpu_convolution(padded_image_.data(), &padded_image_dims_[0], 
  			  vertical_kernel_.data(),&kernel_dims_[0],
  			  1);

  mvn::range axis_subrange = mvn::range(halfKernel,halfKernel+imageDimSize);
  image_ = padded_image_[ boost::indices[axis_subrange][axis_subrange][axis_subrange]];
  float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}

BOOST_AUTO_TEST_CASE( depth_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_depth_.origin(), image_folded_by_depth_.origin() + image_folded_by_depth_.num_elements(),0.f);
  inplace_cpu_convolution(padded_image_.data(), &padded_image_dims_[0], 
  			  depth_kernel_.data(),&kernel_dims_[0],
  			  1);

  mvn::range axis_subrange = mvn::range(halfKernel,halfKernel+imageDimSize);
  image_ = padded_image_[ boost::indices[axis_subrange][axis_subrange][axis_subrange]];
  float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}

BOOST_AUTO_TEST_CASE( all1_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_all1_.origin(), image_folded_by_all1_.origin() + image_folded_by_all1_.num_elements(),0.f);
  inplace_cpu_convolution(padded_image_.data(), &padded_image_dims_[0], 
  			  all1_kernel_.data(),&kernel_dims_[0],
  			  1);

  mvn::range axis_subrange = mvn::range(halfKernel,halfKernel+imageDimSize);
  image_ = padded_image_[ boost::indices[axis_subrange][axis_subrange][axis_subrange]];
  float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( parallel_convolution_works, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( horizontal_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_horizontal_.origin(), image_folded_by_horizontal_.origin() + image_folded_by_horizontal_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  horizont_kernel_.data(),&kernel_dims_[0],
  			  boost::thread::hardware_concurrency()/2);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum, sum_original, .00001f);

}

BOOST_AUTO_TEST_CASE( vertical_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_vertical_.origin(), image_folded_by_vertical_.origin() + image_folded_by_vertical_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  vertical_kernel_.data(),&kernel_dims_[0],
  			  boost::thread::hardware_concurrency()/2);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}

BOOST_AUTO_TEST_CASE( depth_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_depth_.origin(), image_folded_by_depth_.origin() + image_folded_by_depth_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  depth_kernel_.data(),&kernel_dims_[0],
  			  boost::thread::hardware_concurrency()/2);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}

BOOST_AUTO_TEST_CASE( all1_convolve )
{
  

  float sum_original = std::accumulate(image_folded_by_all1_.origin(), image_folded_by_all1_.origin() + image_folded_by_all1_.num_elements(),0.f);
  inplace_cpu_convolution(image_.data(), &image_dims_[0], 
  			  all1_kernel_.data(),&kernel_dims_[0],
  			  boost::thread::hardware_concurrency()/2);

  float sum = std::accumulate(image_.origin(), image_.origin() + image_.num_elements(),0.f);
  BOOST_CHECK_CLOSE(sum, sum_original, .00001);


}
BOOST_AUTO_TEST_SUITE_END()

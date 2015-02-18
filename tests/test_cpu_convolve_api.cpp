#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE HALF_CPU_CONVOLVE_API
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <iostream>
//#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
#include "test_algorithms.hpp"
#include "image_stack_utils.h"

namespace mvn = multiviewnative;

typedef mvn::zero_padd<mvn::image_stack> zero_padding;
static mvn::storage local_order = boost::c_storage_order();

BOOST_FIXTURE_TEST_SUITE( cpu_convolve, mvn::default_3D_fixture )
BOOST_AUTO_TEST_CASE( image_only_constructor )
{
  
  mvn::cpu_convolve<> fold(padded_image_.data(), &padded_image_dims_[0], &kernel_dims_[0]);
  BOOST_CHECK(fold.kernel_ == nullptr);
  BOOST_CHECK(fold.image_ != nullptr);
  BOOST_CHECK_EQUAL_COLLECTIONS(fold.image_->data(), fold.image_->data() + padded_image_.num_elements(),
				padded_image_.data(), padded_image_.data() + padded_image_.num_elements());

  BOOST_CHECK(std::equal(fold.extents_.begin(),fold.extents_.end(), padded_image_dims_.begin()));
  BOOST_CHECK(!std::equal(fold.fft.fftw_shape()->begin(),fold.fft.fftw_shape()->end(), padded_image_dims_.cbegin()));
}

BOOST_AUTO_TEST_CASE( constructor )
{
  
mvn::cpu_convolve<> fold(padded_image_.data(), &padded_image_dims_[0], 
			   identity_kernel_.data(),&kernel_dims_[0]);

  BOOST_CHECK(fold.kernel_ != nullptr);
  BOOST_CHECK(fold.image_ != nullptr);
  BOOST_CHECK_EQUAL_COLLECTIONS(fold.image_->data(), fold.image_->data() + padded_image_.num_elements(),
				padded_image_.data(), padded_image_.data() + padded_image_.num_elements());

  BOOST_CHECK(std::equal(fold.extents_.begin(),fold.extents_.end(), padded_image_dims_.begin()));
  BOOST_CHECK(!std::equal(fold.fft.fftw_shape()->begin(),fold.fft.fftw_shape()->end(), padded_image_dims_.cbegin()));
}

BOOST_AUTO_TEST_CASE( half_inplace_throws_due_to_wrong_kernel_setup )
{

  //setup kernel
  std::vector<unsigned> padded_kernel_dims(kernel_dims_.begin(), kernel_dims_.end());
  padded_kernel_dims[2] = 2*(padded_kernel_dims[2]/2 + 1);
  mvn::fftw_image_stack padded_kernel(padded_kernel_dims);
  std::fill(padded_kernel.data(), padded_kernel.data() + padded_kernel.num_elements(),0);
  
  mvn::cpu_convolve<> fold(padded_image_.data(), &padded_image_dims_[0], &kernel_dims_[0]);
  BOOST_CHECK_THROW(fold.half_inplace(padded_kernel), std::length_error);

}

BOOST_AUTO_TEST_SUITE_END()

//using namespace mvn;
using mvn::operator<<;

BOOST_FIXTURE_TEST_SUITE( half_convolution_works, mvn::default_3D_fixture )


BOOST_AUTO_TEST_CASE( identity_convolve )
{
  
  mvn::fftw_image_stack expected = padded_image_;
  float sum_original = std::accumulate(padded_image_.data(), padded_image_.data() + padded_image_.num_elements(),0.f);
  
  //prepare kernel
  mvn::fftw_image_stack padded_kernel(padded_image_dims_);
  mvn::cpu_convolve<>::padding_policy padder(&padded_image_dims_[0],&kernel_dims_[0]);
  padder.wrapped_insert_at_offsets(identity_kernel_, padded_kernel);

  //fft it here
  mvn::cpu_convolve<>::transform_policy fft(padded_image_dims_);
  fft.padd_for_fft(&padded_kernel);
  fft.forward(&padded_kernel);
  
  //perform convolution
  mvn::cpu_convolve<> fold(padded_image_.data(), &padded_image_dims_[0], &kernel_dims_[0]);
  fold.half_inplace(padded_kernel);

  
  float sum = std::accumulate(padded_image_.data(), padded_image_.data() + padded_image_.num_elements(),0.f);

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
  
  //prepare kernel
  mvn::fftw_image_stack padded_kernel(padded_image_dims_);
  mvn::cpu_convolve<>::padding_policy padder(&padded_image_dims_[0],&kernel_dims_[0]);
  padder.wrapped_insert_at_offsets(horizont_kernel_, padded_kernel);

  //fft it here
  mvn::cpu_convolve<>::transform_policy fft(padded_image_dims_);
  fft.padd_for_fft(&padded_kernel);
  fft.forward(&padded_kernel);
  
  //perform convolution
  mvn::cpu_convolve<> fold(padded_image_.data(), &padded_image_dims_[0], &kernel_dims_[0]);
  fold.half_inplace(padded_kernel);

  mvn::fftw_image_stack result = padded_image_[ boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]]];
  
  float sum = std::accumulate(result.data(), result.data() + result.num_elements(),0.f);

  try{
    BOOST_REQUIRE_CLOSE(sum, sum_original, .0001);
  }
  catch(...){

    std::cout << "received:" << result << "\n\n"
    	      << "expected:" << image_folded_by_horizontal_ << "\n\n";
  }


}



BOOST_AUTO_TEST_CASE( vertical_convolve )
{
    
  float sum_original = std::accumulate(image_folded_by_vertical_.origin(), image_folded_by_vertical_.origin() + image_folded_by_vertical_.num_elements(),0.f);
  
  //prepare kernel
  mvn::fftw_image_stack padded_kernel(padded_image_dims_);
  mvn::cpu_convolve<>::padding_policy padder(&padded_image_dims_[0],&kernel_dims_[0]);
  padder.wrapped_insert_at_offsets(vertical_kernel_, padded_kernel);

  //fft it here
  mvn::cpu_convolve<>::transform_policy fft(padded_image_dims_);
  fft.padd_for_fft(&padded_kernel);
  fft.forward(&padded_kernel);
  
  //perform convolution
  mvn::cpu_convolve<> fold(padded_image_.data(), &padded_image_dims_[0], &kernel_dims_[0]);
  fold.half_inplace(padded_kernel);

  mvn::fftw_image_stack result = padded_image_[ boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]]];
  
  float sum = std::accumulate(result.data(), result.data() + result.num_elements(),0.f);

  try{
    BOOST_REQUIRE_CLOSE(sum, sum_original, .0001);
  }
  catch(...){

    std::cout << "received:" << result << "\n\n"
    	      << "expected:" << image_folded_by_vertical_ << "\n\n";
  }


}


BOOST_AUTO_TEST_CASE( depth_convolve )
{
    
  float sum_original = std::accumulate(image_folded_by_depth_.origin(), image_folded_by_depth_.origin() + image_folded_by_depth_.num_elements(),0.f);
  
  //prepare kernel
  mvn::fftw_image_stack padded_kernel(padded_image_dims_);
  mvn::cpu_convolve<>::padding_policy padder(&padded_image_dims_[0],&kernel_dims_[0]);
  padder.wrapped_insert_at_offsets(depth_kernel_, padded_kernel);

  //fft it here
  mvn::cpu_convolve<>::transform_policy fft(padded_image_dims_);
  fft.padd_for_fft(&padded_kernel);
  fft.forward(&padded_kernel);
  
  //perform convolution
  mvn::cpu_convolve<> fold(padded_image_.data(), &padded_image_dims_[0], &kernel_dims_[0]);
  fold.half_inplace(padded_kernel);

  mvn::fftw_image_stack result = padded_image_[ boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]]];
  
  float sum = std::accumulate(result.data(), result.data() + result.num_elements(),0.f);

  try{
    BOOST_REQUIRE_CLOSE(sum, sum_original, .0001);
  }
  catch(...){

    std::cout << "received:" << result << "\n\n"
    	      << "expected:" << image_folded_by_depth_ << "\n\n";
  }


}

BOOST_AUTO_TEST_SUITE_END()



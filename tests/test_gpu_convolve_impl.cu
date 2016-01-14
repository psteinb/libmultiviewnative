#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_CONVOLUTION_IMPL
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"

#include <numeric>

#include "multiviewnative.h"
#include "padd_utils.h"

#include "cuda_memory.cuh"
#include "gpu_convolve.cuh"


using device_stack = multiviewnative::stack_on_device<multiviewnative::image_stack> ;
using image_stack = multiviewnative::image_stack;
using wrap_around_padding = multiviewnative::zero_padd<image_stack>;
using device_transform = multiviewnative::inplace_3d_transform_on_device<float> ;


BOOST_FIXTURE_TEST_SUITE(plain_with_cuda_memory,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(horizontal_convolve) {
  
  float sum_original = std::accumulate(
      image_folded_by_horizontal_.data(),
      image_folded_by_horizontal_.data() + image_.num_elements(), 0.f);

  std::vector<unsigned> cufft_inplace_extents(image_dims_.size());

  wrap_around_padding padding(&image_dims_[0],&kernel_dims_[0]);
  multiviewnative::adapt_extents_for_fftw_inplace(padding.extents_,
						  cufft_inplace_extents,
						  padded_image_.storage_order());

  size_t device_memory_elements_required = std::accumulate(cufft_inplace_extents.begin(),
							   cufft_inplace_extents.end(),
							   1,
							   std::multiplies<size_t>());

  multiviewnative::image_stack padded_kernel = padded_image_;
  std::fill(padded_kernel.data(), padded_kernel.data()+padded_kernel.num_elements(),0);
  padding.wrapped_insert_at_offsets(horizont_kernel_, padded_kernel);

  padded_image_.resize(cufft_inplace_extents);
  padded_kernel.resize(cufft_inplace_extents);
  
  device_stack d_padded_image(padded_image_,device_memory_elements_required);
  device_stack d_padded_kernel(padded_kernel,device_memory_elements_required);
  
  multiviewnative::inplace_convolve_on_device<device_transform>(d_padded_image.data(),
								d_padded_kernel.data(),
								&padding.extents_[0],
								device_memory_elements_required);
  HANDLE_LAST_ERROR();

  d_padded_image.pull_from_device(padded_image_);
  
  image_stack result = padded_image_[boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]]];

  float sum = std::accumulate(result.data(),
                              result.data() + result.num_elements(), 0.f);

  try {
    BOOST_REQUIRE_CLOSE(sum, sum_original, .15);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << "expected:\n";
    multiviewnative::print_stack(image_folded_by_horizontal_);
    std::cout << "\n"
	      << "received:\n";
    multiviewnative::print_stack(result);
    std::cout << "\n"	      // << "padded received:\n"
      ;

    
    // multiviewnative::print_stack(padded_one_);
    std::cout << "\n";
  }

}

BOOST_AUTO_TEST_CASE(horizontal_convolve_from_one) {
  
  float sum_expected = std::accumulate(
      horizont_kernel_.data(),
      horizont_kernel_.data() + horizont_kernel_.num_elements(), 0.f);

  std::vector<unsigned> cufft_inplace_extents(image_dims_.size());

  wrap_around_padding padding(&image_dims_[0],&kernel_dims_[0]);
  multiviewnative::adapt_extents_for_fftw_inplace(padding.extents_,
						  cufft_inplace_extents,
						  padded_one_.storage_order());

  size_t device_memory_elements_required = std::accumulate(cufft_inplace_extents.begin(),
							   cufft_inplace_extents.end(),
							   1,
							   std::multiplies<size_t>());

  multiviewnative::image_stack padded_kernel = padded_one_;
  std::fill(padded_kernel.data(), padded_kernel.data()+padded_kernel.num_elements(),0);
  padding.wrapped_insert_at_offsets(horizont_kernel_, padded_kernel);

  padded_one_.resize(cufft_inplace_extents);
  padded_image_.resize(cufft_inplace_extents);
  padded_kernel.resize(cufft_inplace_extents);
  
  device_stack d_padded_one(padded_one_,device_memory_elements_required);
  device_stack d_padded_kernel(padded_kernel,device_memory_elements_required);
  
  multiviewnative::inplace_convolve_on_device<device_transform>(d_padded_one.data(),
								d_padded_kernel.data(),
								&padding.extents_[0],
								device_memory_elements_required);
  HANDLE_LAST_ERROR();

  d_padded_one.pull_from_device(padded_image_);
  
  image_stack result = padded_image_[boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]]];

  float sum = std::accumulate(result.data(),
                              result.data() + result.num_elements(), 0.f);

  try {
    BOOST_REQUIRE_CLOSE(sum, sum_expected, .15);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << "expected:\n";
    multiviewnative::print_stack(one_);
    std::cout << "\n"
	      << "received:\n";
    multiviewnative::print_stack(result);
    std::cout << "\n";
  }

}



BOOST_AUTO_TEST_CASE(all1_convolve) {
  
  float sum_original = std::accumulate(
      image_folded_by_all1_.data(),
      image_folded_by_all1_.data() + image_.num_elements(), 0.f);

  std::vector<unsigned> cufft_inplace_extents(image_dims_.size());

  wrap_around_padding padding(&image_dims_[0],&kernel_dims_[0]);
  multiviewnative::adapt_extents_for_fftw_inplace(padding.extents_,
						  cufft_inplace_extents,
						  padded_image_.storage_order());

  size_t device_memory_elements_required = std::accumulate(cufft_inplace_extents.begin(),
							   cufft_inplace_extents.end(),
							   1,
							   std::multiplies<size_t>());

  multiviewnative::image_stack padded_kernel = padded_image_;
  std::fill(padded_kernel.data(), padded_kernel.data()+padded_kernel.num_elements(),0);
  padding.wrapped_insert_at_offsets(all1_kernel_, padded_kernel);

  padded_one_.resize(cufft_inplace_extents);
  padded_image_.resize(cufft_inplace_extents);
  padded_kernel.resize(cufft_inplace_extents);
  
  device_stack d_padded_image(padded_image_,device_memory_elements_required);
  device_stack d_padded_kernel(padded_kernel,device_memory_elements_required);
  
  multiviewnative::inplace_convolve_on_device<device_transform>(d_padded_image.data(),
								d_padded_kernel.data(),
								&padding.extents_[0],
								device_memory_elements_required);
  HANDLE_LAST_ERROR();

  d_padded_image.pull_from_device(padded_one_);
  
  // inplace_gpu_convolution(padded_image_.data(), 
  // 			  &padded_image_dims_[0],
  //                         all1_kernel_.data(), 
  // 			  &kernel_dims_[0],
  //                         selectDeviceWithHighestComputeCapability());

  image_stack result = padded_one_[boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]]];

  float sum = std::accumulate(result.data(),
                              result.data() + result.num_elements(), 0.f);

  try {
    BOOST_REQUIRE_CLOSE(sum, sum_original, .00001);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << "expected:\n";
    multiviewnative::print_stack(image_folded_by_all1_);
    std::cout << "\n"
	      << "received:\n";
    multiviewnative::print_stack(result);
    std::cout << "\n";
  }

}
BOOST_AUTO_TEST_SUITE_END()

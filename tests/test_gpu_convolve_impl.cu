#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_CONVOLUTION_IMPL
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"

#include <numeric>

#include "multiviewnative.h"
#include "padd_utils.h"

#include "cuda_memory.cuh"
#include "gpu_convolve.cuh"
#include "tests_config.h"

using device_stack = multiviewnative::stack_on_device<multiviewnative::image_stack> ;
using image_stack = multiviewnative::image_stack;
using wrap_around_padding = multiviewnative::zero_padd<image_stack>;
using device_transform = multiviewnative::inplace_3d_transform_on_device<float> ;

namespace mvn = multiviewnative;

BOOST_FIXTURE_TEST_SUITE(with_padding,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(horizontal_convolve) {
  
  float sum_original = std::accumulate(
      image_folded_by_horizontal_.data(),
      image_folded_by_horizontal_.data() + image_.num_elements(), 0.f);

  std::vector<unsigned> cufft_inplace_extents(image_dims_.size());

  wrap_around_padding padding(&image_dims_[0],&kernel_dims_[0]);
  multiviewnative::adapt_extents_for_cufft_inplace(padding.extents_,
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
  multiviewnative::adapt_extents_for_cufft_inplace(padding.extents_,
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
  multiviewnative::adapt_extents_for_cufft_inplace(padding.extents_,
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

BOOST_FIXTURE_TEST_SUITE(no_padding,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(horizontal_convolve) {
  
  float sum_original = std::accumulate(
      image_folded_by_horizontal_.data(),
      image_folded_by_horizontal_.data() + image_.num_elements(), 0.f);

  std::vector<unsigned> cufft_inplace_extents(image_dims_.size());

  multiviewnative::no_padd<multiviewnative::image_stack> padding(&padded_image_dims_[0],&kernel_dims_[0]);
  multiviewnative::adapt_extents_for_cufft_inplace(padding.extents_,
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
  padded_image_.resize(padding.extents_);
  
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

BOOST_AUTO_TEST_CASE(all1_convolve) {
  
  float sum_original = std::accumulate(
      image_folded_by_all1_.data(),
      image_folded_by_all1_.data() + image_.num_elements(), 0.f);

  std::vector<unsigned> cufft_inplace_extents(image_dims_.size());

  multiviewnative::no_padd<multiviewnative::image_stack> padding(&padded_image_dims_[0],&kernel_dims_[0]);
  multiviewnative::adapt_extents_for_cufft_inplace(padding.extents_,
						  cufft_inplace_extents,
						  padded_image_.storage_order());

  size_t device_memory_elements_required = std::accumulate(cufft_inplace_extents.begin(),
							   cufft_inplace_extents.end(),
							   1,
							   std::multiplies<size_t>());

  multiviewnative::image_stack padded_kernel = padded_image_;
  std::fill(padded_kernel.data(), padded_kernel.data()+padded_kernel.num_elements(),0);
  padding.wrapped_insert_at_offsets(all1_kernel_, padded_kernel);

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
  padded_image_.resize(padding.extents_);
  
  image_stack result = padded_image_[boost::indices[symm_ranges_[0]][symm_ranges_[1]][symm_ranges_[2]]];

  float sum = std::accumulate(result.data(),
                              result.data() + result.num_elements(), 0.f);

  try {
    BOOST_REQUIRE_CLOSE(sum, sum_original, .15);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << "expected:\n";
    multiviewnative::print_stack(image_folded_by_all1_);
    std::cout << "\n"
	      << "received:\n";
    multiviewnative::print_stack(result);
    std::cout << "\n"	      // << "padded received:\n"
      ;

    
    // multiviewnative::print_stack(padded_one_);
    std::cout << "\n";
  }

}
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(asymmetric_content,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(horizontal_convolve) {
  
  float sum_original = std::accumulate(one_folded_by_asymm_cross_kernel_.data(),
				       one_folded_by_asymm_cross_kernel_.data() + one_folded_by_asymm_cross_kernel_.num_elements(), 0.f);

  std::vector<unsigned> cufft_inplace_extents(image_dims_.size());

  wrap_around_padding padding(&image_dims_[0],&asymm_kernel_dims_[0]);
  multiviewnative::adapt_extents_for_cufft_inplace(padding.extents_,
						  cufft_inplace_extents,
						  one_.storage_order());


#ifdef LMVN_TRACE
  for(int i = 0;i<3;++i){
    std::cout << i << ": image = " << image_dims_[i]
	      << "\tkernel = " << asymm_kernel_dims_[i]
	      << "\tpadd.ext = " << padding.extents_[i]
	      << "\tasymm_padd_im = " << asymm_padded_one_.shape()[i]
	      << "\n";
  }
#endif


  
  size_t device_memory_elements_required = std::accumulate(cufft_inplace_extents.begin(),
							   cufft_inplace_extents.end(),
							   1,
							   std::multiplies<size_t>());
  
  asymm_padded_one_.resize(padding.extents_);
  padding.insert_at_offsets(one_,asymm_padded_one_);
  
  multiviewnative::image_stack padded_kernel = asymm_padded_one_;
  std::fill(padded_kernel.data(), padded_kernel.data()+padded_kernel.num_elements(),0);
  padding.wrapped_insert_at_offsets(asymm_cross_kernel_, padded_kernel);

  asymm_padded_one_.resize(cufft_inplace_extents);
  padded_kernel.resize(cufft_inplace_extents);
  
  device_stack d_asymm_padded_one_(asymm_padded_one_,device_memory_elements_required);
  device_stack d_padded_kernel(padded_kernel,device_memory_elements_required);
  
  multiviewnative::inplace_convolve_on_device<device_transform>(d_asymm_padded_one_.data(),
								d_padded_kernel.data(),
								&padding.extents_[0],
								device_memory_elements_required);
  HANDLE_LAST_ERROR();

  d_asymm_padded_one_.pull_from_device(asymm_padded_one_);
  
  image_stack result = asymm_padded_one_[boost::indices[multiviewnative::range(
									       padding.offsets_[0],
									       padding.offsets_[0] + one_.shape()[0])]
					 [multiviewnative::range(
								 padding.offsets_[1],
								 padding.offsets_[1] + one_.shape()[1])]
					 [multiviewnative::range(
								 padding.offsets_[2],
								 padding.offsets_[2] + one_.shape()[2])]];

  float sum = std::accumulate(result.data(),
                              result.data() + result.num_elements(), 0.f);



  try {
    BOOST_REQUIRE_CLOSE(sum, sum_original, .15);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << "expected:\n";
    multiviewnative::print_stack(one_folded_by_asymm_cross_kernel_);
    std::cout << "\n"
	      << "received:\n";
    multiviewnative::print_stack(result);
    std::cout << "\n"	      // << "padded received:\n"
      ;

    
    // multiviewnative::print_stack(padded_one_);
    std::cout << "\n";
  }

}

BOOST_AUTO_TEST_CASE(asymm_image) {

  std::vector<unsigned> shape(3,16);
  shape[1] += 2;
  shape[2] -= 2;
  
  std::vector<unsigned> cufft_inplace_extents(shape.size());

  wrap_around_padding padding((int*)&shape[0],&kernel_dims_[0]);
  mvn::adapt_extents_for_cufft_inplace(padding.extents_,
						  cufft_inplace_extents,
						  one_.storage_order());
  image_stack image(shape);

  std::iota(image.data(),image.data()+image.num_elements(),0);
  image_stack expected = image;
    
  size_t device_memory_elements_required = std::accumulate(cufft_inplace_extents.begin(),
							   cufft_inplace_extents.end(),
							   1,
							   std::multiplies<size_t>());

  mvn::image_stack padded_image(padding.extents_);
  std::fill(padded_image.data(), padded_image.data()+padded_image.num_elements(),0);
  mvn::image_stack padded_kernel = padded_image;
  
  padding.insert_at_offsets(image, padded_image);
  padding.wrapped_insert_at_offsets(identity_kernel_, padded_kernel);

  padded_image.resize(cufft_inplace_extents);
  padded_kernel.resize(cufft_inplace_extents);
  
  

  #ifdef LMVN_TRACE
  for(int i = 0;i<3;++i){
    std::cout << i << ": image = " << shape[i]
	      << "\tkernel = " << kernel_dims_[i]
	      << "\tpadd.ext = " << padding.extents_[i]
	      << "\tcufft = " << cufft_inplace_extents[i]
	      << "\tpadd.off = " << padding.offsets_[i]
	      << "\n";
  }

  // std::cout << "padded_image:\n";
  // mvn::print_stack(padded_image);
  // std::cout << "\n\npadded_image_for_cufft:\n";
  // mvn::print_stack(padded_image_for_cufft);
  // std::cout << "\n";
#endif
  
  device_stack d_image(padded_image,
		       //padded_image_for_cufft,
		       device_memory_elements_required);
  device_stack d_padded_kernel(padded_kernel,//padded_kernel_for_cufft,
			       device_memory_elements_required);
  
  mvn::inplace_convolve_on_device<device_transform>(d_image.data(),
								d_padded_kernel.data(),
								&padding.extents_[0],
								device_memory_elements_required);
  HANDLE_LAST_ERROR();



  d_image.pull_from_device(padded_image
			   //padded_image_for_cufft
			   );

  //undo cufft padding
  // image_stack temp_gpu_result(padding.extents_);
  // std::fill(temp_gpu_result.data(), temp_gpu_result.data()+temp_gpu_result.num_elements(),0);
  
  mvn::image_stack_view temp_gpu_result = padded_image// _for_cufft
    [boost::indices[mvn::range(0,padding.extents_[0])]
     [mvn::range(0,padding.extents_[1])]
     [mvn::range(0,padding.extents_[2])]
     ];
  
  image_stack gpu_result = temp_gpu_result[boost::indices[mvn::range(
										 padding.offsets_[0],
										 padding.offsets_[0] + shape[0])]
					   [mvn::range(
								   padding.offsets_[1],
								   padding.offsets_[1] + shape[1])]
					   [mvn::range(
								   padding.offsets_[2],
								   padding.offsets_[2] + shape[2])]];

  for(size_t d = 0;d<3;++d)
    BOOST_CHECK_EQUAL(shape[d],gpu_result.shape()[d]);
  
  for(size_t p = 0;p<gpu_result.num_elements();++p){
    try {
      BOOST_REQUIRE_SMALL(std::abs(gpu_result.data()[p]-image.data()[p]),0.001f);
    }
    catch(...){
      std::cout << "gpu_result differs "<< gpu_result.data()[p] << " != " << image.data()[p] <<" at pixel " << p << " / " << gpu_result.num_elements() << "\n\nexpected:\n";
      
      mvn::print_stack(image);
      std::cout << "\n\nreceived:\n";
      mvn::print_stack(gpu_result);
      std::cout << "\n";
      break;
    }
  }
  
  
}
BOOST_AUTO_TEST_SUITE_END()

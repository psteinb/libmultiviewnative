#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_DECONVOLVE_IMPL
#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "gpu_deconvolve_methods.cuh"
#include "convert_tiff_fixtures.hpp"
#include "multiviewnative.h"

#include "test_algorithms.hpp"

using namespace multiviewnative;

typedef multiviewnative::zero_padd<multiviewnative::image_stack>
    as_is_padding;

typedef multiviewnative::inplace_3d_transform_on_device<imageType>
    device_transform;

typedef multiviewnative::gpu_convolve<as_is_padding, imageType, unsigned>
    target_convolve;


static const PaddedReferenceData reference;
static const first_2_iterations two_guesses;
static const first_5_iterations five_guesses;
static const all_iterations all_guesses;

BOOST_AUTO_TEST_SUITE(interleaved)

BOOST_AUTO_TEST_CASE(runs_at_all) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(two_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 2;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);

  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack reference = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack expected = *local_guesses.padded_psi(2,shape_to_padd_with);

  // test
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_interleaved<as_is_padding, 
					       target_convolve,
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  BOOST_CHECK(gpu_input_psi != reference);

  delete [] input.data_;
}

BOOST_AUTO_TEST_CASE(l2norm) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(two_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 2;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);

  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack reference = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack expected = *local_guesses.padded_psi(0,shape_to_padd_with);
  inplace_cpu_deconvolve(expected.data(), input, -1);
  
  // test
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_interleaved<as_is_padding, 
					       target_convolve,
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  // check norms
  const float bottom_ratio = .25;
  const float upper_ratio = .75;
  float l2norm_to_guesses = multiviewnative::l2norm_within_limits(gpu_input_psi, expected,
								  bottom_ratio,
								  upper_ratio);
  BOOST_CHECK_LT(l2norm_to_guesses, 1);
  if(l2norm_to_guesses>10){
    multiviewnative::write_image_stack(gpu_input_psi,"test_gpu_deconvolve_impl_interleaved_l2norm.tiff");
  }
  delete [] input.data_;
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(all_on_device)

BOOST_AUTO_TEST_CASE(runs_at_all) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(two_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 2;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);

  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack reference = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack expected = *local_guesses.padded_psi(2,shape_to_padd_with);

  // test
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_all_on_device<as_is_padding, 
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  BOOST_CHECK(gpu_input_psi != reference);

  for(int i = 0;i<3;++i)
    BOOST_CHECK_EQUAL(gpu_input_psi.shape()[i],expected.shape()[i]);

  
  delete [] input.data_;
}

BOOST_AUTO_TEST_CASE(l2norm_after_2_guesses) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(two_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 2;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);
  image_stack reference = *local_guesses.psi(0);
  
#ifdef LMVN_TRACE
  std::cout << "[trace::"<< __FILE__ <<"] padding reference data ";
  for(int im = 0;im<3;++im)
    std::cout << reference.shape()[im] << " ";
  std::cout << " with ";
  for( int ex : shape_to_padd_with)
    std::cout << ex << " ";
  std::cout << "\n";
#endif
  
  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  
  image_stack expected = *local_guesses.padded_psi(0,shape_to_padd_with);

  // test gpu
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_all_on_device<as_is_padding, 
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  for(int i = 0;i<3;++i){
    BOOST_CHECK_EQUAL(gpu_input_psi.shape()[i],expected.shape()[i]);
  }
  
  //run cpu
  inplace_cpu_deconvolve(expected.data(), input, -1);
  std::array<range,image_stack::dimensionality> padding_offset = *local_guesses.offset(0);
  
  image_stack cpu_result = expected[boost::indices[padding_offset[0]][padding_offset[1]][padding_offset[2]]];
  image_stack gpu_result = gpu_input_psi[boost::indices[padding_offset[0]][padding_offset[1]][padding_offset[2]]];
  
  // check norms
  float l2norm_to_cpu 	 = multiviewnative::l2norm(cpu_result.data(),gpu_result.data(),reference.num_elements());
  float l2norm_to_ref 	 = multiviewnative::l2norm(reference .data(),gpu_result.data(),reference.num_elements());
  float l2norm_to_cpu_ref= multiviewnative::l2norm(cpu_result.data(),reference .data(),reference.num_elements());

  #ifdef LMVN_TRACE
  std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
	    << " l2norm(gpu-cpu) = " << l2norm_to_cpu	  << " /size = \t" << l2norm_to_cpu    /double(reference.num_elements()) << "\n" 
	    << " l2norm(gpu-ref) = " << l2norm_to_ref	  << " /size = \t" << l2norm_to_ref    /double(reference.num_elements()) << "\n"
	    << " l2norm(cpu-ref) = " << l2norm_to_cpu_ref << " /size = \t" << l2norm_to_cpu_ref/double(reference.num_elements()) << "\n"
    ;
  #endif
  
  BOOST_CHECK_LT(l2norm_to_cpu, 1);

  if(l2norm_to_cpu>1){
    multiviewnative::write_image_stack(gpu_result,"all_on_device_gpu_psi_2.tiff");
    multiviewnative::write_image_stack(cpu_result,"all_on_device_cpu_psi_2.tiff");
    multiviewnative::write_image_stack(reference,"all_on_device_ref_psi_2.tiff");
  }
  delete [] input.data_;
}


BOOST_AUTO_TEST_CASE(l2norm_after_5_guesses) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_5_iterations local_guesses(five_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 5;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);
  
  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack reference = *local_guesses.psi(0);
  image_stack expected = *local_guesses.padded_psi(0,shape_to_padd_with);

  // test gpu
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_all_on_device<as_is_padding, 
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  for(int i = 0;i<3;++i){
    BOOST_CHECK_EQUAL(gpu_input_psi.shape()[i],expected.shape()[i]);
  }
  
  //run cpu
  inplace_cpu_deconvolve(expected.data(), input, -1);
  std::array<range,image_stack::dimensionality> padding_offset = *local_guesses.offset(0);
  
  image_stack cpu_result = expected[boost::indices[padding_offset[0]][padding_offset[1]][padding_offset[2]]];
  image_stack gpu_result = gpu_input_psi[boost::indices[padding_offset[0]][padding_offset[1]][padding_offset[2]]];
  
  // check norms
  float l2norm_to_cpu 	 = multiviewnative::l2norm(cpu_result.data(),gpu_result.data(),reference.num_elements());
  float l2norm_to_ref 	 = multiviewnative::l2norm(reference .data(),gpu_result.data(),reference.num_elements());
  float l2norm_to_cpu_ref= multiviewnative::l2norm(cpu_result.data(),reference .data(),reference.num_elements());

  #ifdef LMVN_TRACE
  std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
	    << " l2norm(gpu-cpu) = " << l2norm_to_cpu	  << " /size = \t" << l2norm_to_cpu    /double(reference.num_elements()) << "\n" 
	    << " l2norm(gpu-ref) = " << l2norm_to_ref	  << " /size = \t" << l2norm_to_ref    /double(reference.num_elements()) << "\n"
	    << " l2norm(cpu-ref) = " << l2norm_to_cpu_ref << " /size = \t" << l2norm_to_cpu_ref/double(reference.num_elements()) << "\n"
    ;
  #endif
  
  BOOST_CHECK_LT(l2norm_to_cpu, 1);

  if(l2norm_to_cpu>1){
    multiviewnative::write_image_stack(gpu_result,"all_on_device_gpu_psi_5.tiff");
    multiviewnative::write_image_stack(cpu_result,"all_on_device_cpu_psi_5.tiff");
    multiviewnative::write_image_stack(reference, "all_on_device_ref_psi_5.tiff");
  }
  delete [] input.data_;

}



BOOST_AUTO_TEST_CASE(l2norm_after_10_guesses) {


  // setup
  PaddedReferenceData local_ref(reference);
  all_iterations local_guesses(all_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 10;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);
  
  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack reference = *local_guesses.psi(0);
  image_stack expected = *local_guesses.padded_psi(0,shape_to_padd_with);

  // test gpu
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_all_on_device<as_is_padding, 
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  for(int i = 0;i<3;++i){
    BOOST_CHECK_EQUAL(gpu_input_psi.shape()[i],expected.shape()[i]);
  }
  
  //run cpu
  inplace_cpu_deconvolve(expected.data(), input, -1);
  std::array<range,image_stack::dimensionality> padding_offset = *local_guesses.offset(0);
  
  image_stack cpu_result = expected[boost::indices[padding_offset[0]][padding_offset[1]][padding_offset[2]]];
  image_stack gpu_result = gpu_input_psi[boost::indices[padding_offset[0]][padding_offset[1]][padding_offset[2]]];
  
  // check norms
  float l2norm_to_cpu 	 = multiviewnative::l2norm(cpu_result.data(),gpu_result.data(),reference.num_elements());
  float l2norm_to_ref 	 = multiviewnative::l2norm(reference .data(),gpu_result.data(),reference.num_elements());
  float l2norm_to_cpu_ref= multiviewnative::l2norm(cpu_result.data(),reference .data(),reference.num_elements());

  #ifdef LMVN_TRACE
  std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
	    << " l2norm(gpu-cpu) = " << l2norm_to_cpu	  << " /size = \t" << l2norm_to_cpu    /double(reference.num_elements()) << "\n" 
	    << " l2norm(gpu-ref) = " << l2norm_to_ref	  << " /size = \t" << l2norm_to_ref    /double(reference.num_elements()) << "\n"
	    << " l2norm(cpu-ref) = " << l2norm_to_cpu_ref << " /size = \t" << l2norm_to_cpu_ref/double(reference.num_elements()) << "\n"
    ;
  #endif
  
  BOOST_CHECK_LT(l2norm_to_cpu, 1);

  if(l2norm_to_cpu>1){
    multiviewnative::write_image_stack(gpu_result,"all_on_device_gpu_psi_9.tiff");
    multiviewnative::write_image_stack(cpu_result,"all_on_device_cpu_psi_9.tiff");
    multiviewnative::write_image_stack(reference,"all_on_device_ref_psi_9.tiff");
  }
  delete [] input.data_;
}

BOOST_AUTO_TEST_CASE(l2norm_after_0_guesses) {


  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(two_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 0;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);
  
  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack reference = *local_guesses.psi(0);

  // test gpu
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_all_on_device<as_is_padding, 
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  

  std::array<range,image_stack::dimensionality> padding_offset = *local_guesses.offset(0);
  
  image_stack gpu_result = gpu_input_psi[boost::indices[padding_offset[0]][padding_offset[1]][padding_offset[2]]];
  
  // check norms
  float l2norm_to_ref 	 = multiviewnative::l2norm(reference .data(),gpu_result.data(),reference.num_elements());

  #ifdef LMVN_TRACE
  std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
	    << " l2norm(gpu-ref) = " << l2norm_to_ref	  << " /size = \t" << l2norm_to_ref    /double(reference.num_elements()) << "\n"
    ;
  multiviewnative::write_image_stack(gpu_result,"all_on_device_gpu_psi_0.tiff");
  multiviewnative::write_image_stack(reference,"all_on_device_ref_psi_0.tiff");

  #endif
  
  BOOST_CHECK_LT(l2norm_to_ref, 1);
  delete [] input.data_;
}

BOOST_AUTO_TEST_SUITE_END()

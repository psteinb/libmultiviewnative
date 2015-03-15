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
static const first_2_iterations five_guesses;

BOOST_AUTO_TEST_SUITE(interleaved)

BOOST_AUTO_TEST_CASE(runs_at_all) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(five_guesses);
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

  // test
  int device_id = selectDeviceWithHighestComputeCapability();
  inplace_gpu_deconvolve_iteration_interleaved<as_is_padding, 
					       target_convolve,
					       device_transform>
    (gpu_input_psi.data(), input, device_id);

  BOOST_CHECK(gpu_input_psi != reference);
  
  delete[] input.data_;
}
BOOST_AUTO_TEST_SUITE_END()

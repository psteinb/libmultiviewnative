#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_DECONVOLVE
#include <sstream>
#include <chrono>

#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"
#include "convert_tiff_fixtures.hpp"

// #include "cpu_convolve.h"
// #include "padd_utils.h"
// #include "fft_utils.h"
// #include "cpu_kernels.h"
// #include "test_algorithms.hpp"
#include "logging.hpp"
#include "cuda_helpers.cuh"
#include "synthetic_data.hpp"

using namespace multiviewnative;

static const PaddedReferenceData reference;
static const all_iterations guesses;


BOOST_AUTO_TEST_SUITE(bernchmark)

BOOST_AUTO_TEST_CASE(deconvolve_all_gpus_lambda_6) {
  // setup
  PaddedReferenceData local_ref(reference);
  all_iterations local_guesses(guesses);

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);

  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 10;
  image_stack start_psi = *local_guesses.padded_psi(0, shape_to_padd_with);
  std::vector<int> image_shape(start_psi.shape(), start_psi.shape() + 3);
  int num_repeats = 10;
  int device_id = selectDeviceWithHighestComputeCapability();
  //parallel
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0;i < num_repeats;++i)
    inplace_gpu_deconvolve(start_psi.data(), input, device_id);
  auto parallel_time = std::chrono::high_resolution_clock::now() - start;

  std::string comments = "";
std::string device_name = get_cuda_device_name(device_id);
  print_info(1,
	     __FILE__, 
	       device_name,
	     num_repeats,
	     std::chrono::duration_cast<std::chrono::milliseconds>(parallel_time).count(),
	     image_shape,
	     (int)sizeof(float),
	     comments
	     );

  // tear-down
  delete[] input.data_;
}

BOOST_AUTO_TEST_CASE(deconvolve_interleaved_gpus_lambda_6){

  workspace input;
  input.data_ = 0;
  std::vector<int> shape(3,256);
  shape[0] = 512;
  shape[1] = 512;
  multiviewnative::multiview_data syn(shape);
  syn.fill_workspace(input);
  input.num_iterations_ = 10;
  input.lambda_ = .006;
  input.minValue_ = .001;

  image_stack start_psi = syn.views_[0];

  std::vector<int> image_shape(start_psi.shape(), start_psi.shape() + 3);
  int num_repeats = 10;
  int device_id = selectDeviceWithHighestComputeCapability();
  //parallel
  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0;i < num_repeats;++i)
    inplace_gpu_deconvolve(start_psi.data(), input, device_id);
  auto parallel_time = std::chrono::high_resolution_clock::now() - start;

  std::string comments = "";
  std::string device_name = get_cuda_device_name(device_id);
  print_info(1,
	     __FILE__, 
	     device_name,
	     num_repeats,
	     std::chrono::duration_cast<std::chrono::milliseconds>(parallel_time).count(),
	     image_shape,
	     (int)sizeof(float),
	     comments
	     );

  // tear-down
  delete[] input.data_;
}

BOOST_AUTO_TEST_SUITE_END()

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_DECONVOLVE
#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"
#include "convert_tiff_fixtures.hpp"

//#include "cuda_memory.cuh"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
#include "cpu_kernels.h"
#include "test_algorithms.hpp"

using namespace multiviewnative;

static const PaddedReferenceData reference;
static const first_5_iterations five_guesses;

BOOST_AUTO_TEST_SUITE(deconvolve_psi0_on_device)

BOOST_AUTO_TEST_CASE(compare_to_cpu) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_5_iterations local_guesses(five_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 1;

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);

  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);
  image_stack cpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);

  // test
  inplace_gpu_deconvolve(gpu_input_psi.data(), input, -1);
  inplace_cpu_deconvolve(cpu_input_psi.data(), input, 1);

  // check norms
  float l2norm_to_guesses = multiviewnative::l2norm(
      gpu_input_psi.data(), local_guesses.psi(1)->data(),
      gpu_input_psi.num_elements());
  float l2norm_to_cpu = multiviewnative::l2norm(
      gpu_input_psi.data(), cpu_input_psi.data(), gpu_input_psi.num_elements());

  BOOST_CHECK_LT(l2norm_to_cpu, 1);
  BOOST_CHECK_LT(l2norm_to_guesses, 1);
  std::cout << ">>" << boost::unit_test::framework::current_test_case().p_name
            << "<<\n"
            << "l2norm_to_guesses\t" << l2norm_to_guesses << "\nl2norm_to_cpu\t"
            << l2norm_to_cpu << "\n";

  const float bottom_ratio = .25;
  const float upper_ratio = .75;
  l2norm_to_guesses = multiviewnative::l2norm_within_limits(
							    gpu_input_psi, *local_guesses.padded_psi(1,shape_to_padd_with), bottom_ratio, upper_ratio);
  l2norm_to_cpu = multiviewnative::l2norm_within_limits(
      gpu_input_psi, cpu_input_psi, bottom_ratio, upper_ratio);
  // tear-down
  BOOST_CHECK_LT(l2norm_to_cpu, 1e-1);
  BOOST_CHECK_LT(l2norm_to_guesses, 1e-1);

  unsigned prec = std::cout.precision();
  std::cout.precision(2);
  std::cout << "[" << bottom_ratio << ", " << upper_ratio << "] ";
  std::cout.precision(prec);
  std::cout << "l2norm_to_guesses\t" << l2norm_to_guesses << "\nl2norm_to_cpu\t"
            << l2norm_to_cpu << "\n";

  delete[] input.data_;
}

BOOST_AUTO_TEST_CASE(compare_to_guesses_after_4_iterations) {
  std::cout << ">>" << boost::unit_test::framework::current_test_case().p_name
            << "<<\n";
  // setup
  PaddedReferenceData local_ref(reference);
  first_5_iterations local_guesses(five_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 4;

  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);
  image_stack gpu_input_psi = *local_guesses.padded_psi(0, shape_to_padd_with);

  // test
  inplace_gpu_deconvolve(gpu_input_psi.data(), input, -1);

  // check norms
  float l2norm_to_guesses = multiviewnative::l2norm(
      gpu_input_psi.data(), local_guesses.padded_psi(5, shape_to_padd_with)->data(),
      gpu_input_psi.num_elements());
  BOOST_CHECK_LT(l2norm_to_guesses, 20);
  std::cout << "l2norm_to_guesses\t" << l2norm_to_guesses << "\n";

  const float bottom_ratio = .25;
  const float upper_ratio = .75;
  l2norm_to_guesses = multiviewnative::l2norm_within_limits(
      gpu_input_psi, *local_guesses.psi(5), bottom_ratio, upper_ratio);
  // tear-down
  BOOST_REQUIRE_LT(l2norm_to_guesses, 1e-1);

  unsigned prec = std::cout.precision();
  std::cout.precision(2);
  std::cout << "[" << bottom_ratio << ", " << upper_ratio << "] ";
  std::cout.precision(prec);
  std::cout << "l2norm_to_guesses\t" << l2norm_to_guesses << "\n";

  delete[] input.data_;
}

BOOST_AUTO_TEST_CASE(run_1) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses;
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 1;

  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);
  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);

  // test
  inplace_gpu_deconvolve(gpu_input_psi.data(), input, -1);

  delete[] input.data_;
}

BOOST_AUTO_TEST_CASE(run_5) {

  // setup
  PaddedReferenceData local_ref(reference);
  first_5_iterations local_guesses(five_guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 5;
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);
  image_stack gpu_input_psi = *local_guesses.padded_psi(0,shape_to_padd_with);

  // test
  inplace_gpu_deconvolve(gpu_input_psi.data(), input, -1);

  delete[] input.data_;
}
BOOST_AUTO_TEST_SUITE_END()

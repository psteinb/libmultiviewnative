#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE GPU_DECONVOLVE
#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"
#include "convert_tiff_fixtures.hpp"

#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
#include "cpu_kernels.h"
#include "test_algorithms.hpp"

using namespace multiviewnative;

static const ReferenceData reference;
static const first_5_iterations guesses;

BOOST_AUTO_TEST_SUITE( deconvolve_psi0_on_device  )

BOOST_AUTO_TEST_CASE( compare_to_cpu )
{

  //setup
  ReferenceData local_ref(reference);
  first_5_iterations local_guesses(guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_, local_guesses.minValue_);
  input.num_iterations_ = 1;
  image_stack gpu_input_psi = *local_guesses.psi(0);
  image_stack cpu_input_psi = *local_guesses.psi(0);

  //test
  inplace_gpu_deconvolve(gpu_input_psi.data(), input, -1);
  inplace_cpu_deconvolve_iteration(cpu_input_psi.data(), input, 2);

  //check norms
  float l2norm_to_guesses = multiviewnative::l2norm(gpu_input_psi.data(), local_guesses.psi(1)->data(), gpu_input_psi.num_elements());
  float l2norm_to_cpu = multiviewnative::l2norm(gpu_input_psi.data(), cpu_input_psi.data(), gpu_input_psi.num_elements());
  try{
    BOOST_REQUIRE_LT(l2norm_to_cpu, 1e-2);
    BOOST_REQUIRE_LT(l2norm_to_guesses, 10);
  }
  catch (...){
    std::cout << "l2norm_to_guesses\t" << l2norm_to_guesses << "\nl2norm_to_cpu\t" << l2norm_to_cpu << "\n";
  }

  //tear-down
  delete [] input.data_;


}

BOOST_AUTO_TEST_SUITE_END()















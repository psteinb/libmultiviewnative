#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE GPU_DECONVOLVE_SINGLE_STEPPED
#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"
#include "convert_tiff_fixtures.hpp"

#include "cuda_memory.cuh"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
#include "cpu_kernels.h"
#include "test_algorithms.hpp"
#include "cuda_kernels.cuh"

using namespace multiviewnative;

// static const ReferenceData reference;
// static const first_5_iterations five_guesses;


BOOST_AUTO_TEST_SUITE( cpu_vs_gpu )
   
BOOST_AUTO_TEST_CASE( divide )
{
  ViewFromDisk view_0(0);


  const unsigned num_elements = view_0.image()->num_elements();

  std::vector<float> cpu_quarter(num_elements);
  std::fill(cpu_quarter.begin(), cpu_quarter.end(), .25);
  std::vector<float> gpu_quarter_results(num_elements,0.f);

  parallel_divide(view_0.image()->data(),&cpu_quarter[0],cpu_quarter.size());

  multiviewnative::stack_on_device<multiviewnative::image_stack> image(*view_0.image());
  multiviewnative::image_stack quarter_stack(*view_0.image());
  std::fill(quarter_stack.data(), quarter_stack.data() + num_elements, .25);
  multiviewnative::image_stack quarter_stack_result(quarter_stack);
  multiviewnative::stack_on_device<multiviewnative::image_stack> quarter_stack_on_device(quarter_stack);
  
  dim3 threads(128);
  dim3 blocks((num_elements + threads.x - 1)/threads.x);
  device_divide<<<blocks,threads>>>(image.data(), 
				    quarter_stack_on_device.data(),
				    num_elements );

  quarter_stack_on_device.pull_from_device(quarter_stack_result);

  float l2norm = multiviewnative::l2norm(quarter_stack_result.data(), &cpu_quarter[0], num_elements);
  BOOST_CHECK_LT(l2norm, 1e-4);
  //  std::cout.precision(4);
  std::cout << "divide\tl2norm = " << l2norm << "\n";
}

BOOST_AUTO_TEST_CASE( final_values )
{
  
  BOOST_FAIL("final_values not implemented yet");
}

BOOST_AUTO_TEST_CASE( regularized_final_values )
{
  
  BOOST_FAIL("final_values not implemented yet");
}

BOOST_AUTO_TEST_SUITE_END()















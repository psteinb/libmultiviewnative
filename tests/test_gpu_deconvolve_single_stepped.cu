#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_DECONVOLVE_SINGLE_STEPPED

// check boost 1.5x and cuda version 7.0 due to bug in c++11 compilation mode
// http://stackoverflow.com/questions/31940457/make-nvcc-output-traces-on-compile-error
#include <boost/predef.h>
#include <cuda.h>
#if defined(BOOST_COMP_GNUC) && BOOST_COMP_GNUC >= BOOST_VERSION_NUMBER(4,7,0) \
   || defined(CUDA_VERSION) && CUDA_VERSION == 7000 \
   || defined(BOOST_VERSION) && BOOST_VERSION >= BOOST_VERSION_NUMBER(1,5,0)
 #include <boost/utility/result_of.hpp>
#endif

#include "boost/test/unit_test.hpp"
#include <functional>
#include <algorithm>

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

#define QUOTE(str) #str
#define STR(str) QUOTE(str)

using namespace multiviewnative;

BOOST_AUTO_TEST_SUITE(cpu_vs_gpu)

BOOST_AUTO_TEST_CASE(divide) {
  ViewFromDisk view_0(0);

  const unsigned num_elements = view_0.image()->num_elements();

  std::vector<float> cpu_quarter(num_elements);
  std::fill(cpu_quarter.begin(), cpu_quarter.end(), .25);
  std::vector<float> gpu_quarter_results(num_elements, 0.f);

  cpu::par::compute_quotient(view_0.image()->data(), &cpu_quarter[0], cpu_quarter.size());

  multiviewnative::stack_on_device<multiviewnative::image_stack> image(
      *view_0.image());
  multiviewnative::stack_on_device<multiviewnative::image_stack> weights(
      *view_0.weights());
  multiviewnative::image_stack quarter_stack(*view_0.image());
  std::fill(quarter_stack.data(), quarter_stack.data() + num_elements, .25);
  multiviewnative::image_stack quarter_stack_result(quarter_stack);
  multiviewnative::stack_on_device<multiviewnative::image_stack>
      quarter_stack_on_device(quarter_stack);

  dim3 threads(128);
  dim3 blocks((num_elements + threads.x - 1) / threads.x);
  device_divide << <blocks, threads>>>
      (image.data(), quarter_stack_on_device.data(), num_elements);

  quarter_stack_on_device.pull_from_device(quarter_stack_result);

  float l2norm = multiviewnative::l2norm(quarter_stack_result.data(),
                                         &cpu_quarter[0], num_elements);
  BOOST_CHECK_LT(l2norm, 1e-4);

  std::fill(quarter_stack.data(), quarter_stack.data() + num_elements,
            1 / 4.01f);
  quarter_stack_on_device.push_to_device(quarter_stack);
  device_divide << <blocks, threads>>>
      (image.data(), quarter_stack_on_device.data(), num_elements);

  quarter_stack_on_device.pull_from_device(quarter_stack_result);
  float l2norm_2 = multiviewnative::l2norm(quarter_stack_result.data(),
                                           &cpu_quarter[0], num_elements);
  BOOST_CHECK_NE(l2norm, l2norm_2);

  std::cout << boost::unit_test::framework::current_test_case().p_name
            << "\tl2norm = " << l2norm << "\n";
  std::cout << boost::unit_test::framework::current_test_case().p_name
            << " (mismatching)\tl2norm = " << l2norm_2 << "\n";
}

BOOST_AUTO_TEST_CASE(final_values) {
  ViewFromDisk view_0(0);

  const unsigned num_elements = view_0.image()->num_elements();

  multiviewnative::image_stack cpu_view_times_2(*view_0.image());
  const float* ibegin = view_0.image()->data();
  const float* iend = ibegin + num_elements;
  float* obegin = cpu_view_times_2.data();

  std::transform(ibegin, iend, ibegin, obegin, std::plus<float>());

  multiviewnative::image_stack gpu_view_times_2(cpu_view_times_2);
  const float minValue = 1e-4;
  cpu::par::final_values(cpu_view_times_2.data(), view_0.image()->data(),
                        view_0.weights()->data(), num_elements, -1, minValue);

  multiviewnative::stack_on_device<multiviewnative::image_stack> d_image(
      *view_0.image());
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_weights(
      *view_0.weights());
  multiviewnative::image_stack gpu_view_times_2_result(gpu_view_times_2);
  multiviewnative::stack_on_device<multiviewnative::image_stack>
      d_gpu_view_times_2(gpu_view_times_2);

  dim3 threads(128);
  dim3 blocks((num_elements + threads.x - 1) / threads.x);
  device_final_values << <blocks, threads>>> (d_gpu_view_times_2.data(),
                                              d_image.data(), d_weights.data(),
                                              minValue, num_elements);

  d_gpu_view_times_2.pull_from_device(gpu_view_times_2_result);

  float l2norm = multiviewnative::l2norm(gpu_view_times_2_result.data(),
                                         cpu_view_times_2.data(), num_elements);
  BOOST_CHECK_LT(l2norm, 1e-4);
  std::cout << boost::unit_test::framework::current_test_case().p_name
            << "\tl2norm = " << l2norm << "\n";
}

BOOST_AUTO_TEST_CASE(regularized_final_values) {
  ViewFromDisk view_0(0);

  const unsigned num_elements = view_0.image()->num_elements();

  multiviewnative::image_stack cpu_view_times_2(*view_0.image());
  const float* ibegin = view_0.image()->data();
  const float* iend = ibegin + num_elements;
  float* obegin = cpu_view_times_2.data();

  std::transform(ibegin, iend, ibegin, obegin, std::plus<float>());

  multiviewnative::image_stack gpu_view_times_2(cpu_view_times_2);
  const float minValue = 1e-4;
  const double lambda = .006;
  cpu::par::regularized_final_values(cpu_view_times_2.data(), view_0.image()->data(), view_0.weights()->data(),
				     num_elements, lambda, -1, minValue);

  multiviewnative::stack_on_device<multiviewnative::image_stack> d_image(
      *view_0.image());
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_weights(
      *view_0.weights());
  multiviewnative::image_stack gpu_view_times_2_result(gpu_view_times_2);
  multiviewnative::stack_on_device<multiviewnative::image_stack>
      d_gpu_view_times_2(gpu_view_times_2);

  dim3 threads(128);
  dim3 blocks((num_elements + threads.x - 1) / threads.x);
  device_regularized_final_values << <blocks, threads>>>
      (d_gpu_view_times_2.data(), d_image.data(), d_weights.data(), lambda,
       minValue, num_elements);

  d_gpu_view_times_2.pull_from_device(gpu_view_times_2_result);

  float l2norm = multiviewnative::l2norm(gpu_view_times_2_result.data(),
                                         cpu_view_times_2.data(), num_elements);
  BOOST_CHECK_LT(l2norm, 1e-4);
  std::cout << boost::unit_test::framework::current_test_case().p_name
            << "\tl2norm = " << l2norm << "\n";
}

BOOST_AUTO_TEST_SUITE_END()

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"

using namespace multiviewnative;
BOOST_FIXTURE_TEST_SUITE(convolution_works_with_asymm_kernels,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(cross_convolve) {

  inplace_cpu_convolution(padded_one_.data(), &padded_image_dims_[0],
                          asymm_cross_kernel_.data(), &asymm_kernel_dims_[0],
                          1);

  range axis_subrange = range(halfKernel, halfKernel + imageDimSize);
  one_ =
      padded_one_[boost::indices[axis_subrange][axis_subrange][axis_subrange]];

  float sum_expected = std::accumulate(
      asymm_cross_kernel_.data(),
      asymm_cross_kernel_.data() + asymm_cross_kernel_.num_elements(), 0.f);
  float sum_received =
      std::accumulate(one_.data(), one_.data() + one_.num_elements(), 0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);

  multiviewnative::range expected_kernel_pos[3];

  for (int i = 0; i < 3; ++i)
    expected_kernel_pos[i] = multiviewnative::range(
        one_.shape()[i] / 2 - asymm_cross_kernel_.shape()[i] / 2,
        one_.shape()[i] / 2 - asymm_cross_kernel_.shape()[i] / 2 +
            asymm_cross_kernel_.shape()[i]);

  multiviewnative::image_stack_view kernel_segment =
      one_[boost::indices[expected_kernel_pos[0]][expected_kernel_pos[1]]
                         [expected_kernel_pos[2]]];

  BOOST_CHECK_EQUAL(kernel_segment.shape()[0], asymm_cross_kernel_.shape()[0]);
  BOOST_CHECK_EQUAL(kernel_segment.shape()[1], asymm_cross_kernel_.shape()[1]);
  BOOST_CHECK_EQUAL(kernel_segment.shape()[2], asymm_cross_kernel_.shape()[2]);

  multiviewnative::image_stack result = kernel_segment;

  for (unsigned p = 0; p < result.num_elements(); ++p)
    BOOST_CHECK_CLOSE_FRACTION(std::floor(result.data()[p] + .5f),
                               asymm_cross_kernel_.data()[p], .1f);
}

BOOST_AUTO_TEST_CASE(one_convolve) {

  inplace_cpu_convolution(padded_one_.data(), &padded_image_dims_[0],
                          asymm_one_kernel_.data(), &asymm_kernel_dims_[0], 1);

  range axis_subrange = range(halfKernel, halfKernel + imageDimSize);
  one_ =
      padded_one_[boost::indices[axis_subrange][axis_subrange][axis_subrange]];

  float sum_expected = std::accumulate(
      asymm_one_kernel_.data(),
      asymm_one_kernel_.data() + asymm_one_kernel_.num_elements(), 0.f);
  float sum_received =
      std::accumulate(one_.data(), one_.data() + one_.num_elements(), 0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);
}

BOOST_AUTO_TEST_CASE(identity_convolve) {

  inplace_cpu_convolution(padded_one_.data(), &padded_image_dims_[0],
                          asymm_identity_kernel_.data(), &asymm_kernel_dims_[0],
                          1);

  range axis_subrange = range(halfKernel, halfKernel + imageDimSize);
  one_ =
      padded_one_[boost::indices[axis_subrange][axis_subrange][axis_subrange]];

  float sum_expected = std::accumulate(
      asymm_identity_kernel_.data(),
      asymm_identity_kernel_.data() + asymm_identity_kernel_.num_elements(),
      0.f);
  float sum_received =
      std::accumulate(one_.data(), one_.data() + one_.num_elements(), 0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);
}

BOOST_AUTO_TEST_CASE(diagonal_convolve) {

  using namespace multiviewnative;
  image_stack diagonal_kernel(asymm_kernel_dims_);
  for (int z_index = 0; z_index < int(diagonal_kernel.shape()[2]); ++z_index) {
    for (int y_index = 0; y_index < int(diagonal_kernel.shape()[1]);
         ++y_index) {
      for (int x_index = 0; x_index < int(diagonal_kernel.shape()[0]);
           ++x_index) {
        if (z_index == y_index && y_index == x_index)
          diagonal_kernel[x_index][y_index][z_index] = 1.f;

        if (z_index == (int(diagonal_kernel.shape()[1]) - 1 - y_index) &&
            y_index == (int(diagonal_kernel.shape()[0]) - 1 - x_index))
          diagonal_kernel[x_index][y_index][z_index] = 1.f;
      }
    }
  }

  inplace_cpu_convolution(padded_one_.data(), &padded_image_dims_[0],
                          diagonal_kernel.data(), &asymm_kernel_dims_[0], 1);

  range axis_subrange = range(halfKernel, halfKernel + imageDimSize);
  one_ =
      padded_one_[boost::indices[axis_subrange][axis_subrange][axis_subrange]];

  float sum_expected = std::accumulate(
      diagonal_kernel.data(),
      diagonal_kernel.data() + diagonal_kernel.num_elements(), 0.f);
  float sum_received =
      std::accumulate(one_.data(), one_.data() + one_.num_elements(), 0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);
}

BOOST_AUTO_TEST_CASE(asymm_one_convolve) {

  inplace_cpu_convolution(
      asymm_padded_one_.data(), &asymm_padded_image_dims_[0],
      asymm_cross_kernel_.data(), &asymm_kernel_dims_[0], 1);

  float sum_expected = std::accumulate(
      asymm_cross_kernel_.data(),
      asymm_cross_kernel_.data() + asymm_cross_kernel_.num_elements(), 0.f);
  float sum_received = std::accumulate(
      asymm_padded_one_.data(),
      asymm_padded_one_.data() + asymm_padded_one_.num_elements(), 0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);
}
BOOST_AUTO_TEST_SUITE_END()

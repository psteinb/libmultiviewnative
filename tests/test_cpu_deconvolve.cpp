#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CPU_DECONVOLVE

#include "boost/test/unit_test.hpp"
#include "boost/test/detail/unit_test_parameters.hpp"
#include "boost/thread.hpp"

#include "tiff_fixtures.hpp"
#include "multiviewnative.h"
#include "convert_tiff_fixtures.hpp"

#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
#include "cpu_kernels.h"
#include "test_algorithms.hpp"

using namespace multiviewnative;

static const PaddedReferenceData reference;
static const first_2_iterations local_guesses_of_2;

BOOST_AUTO_TEST_SUITE(deconvolve)

BOOST_AUTO_TEST_CASE(loaded_data_is_of_same_size) {
  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(local_guesses_of_2);

  for (int i = 1; i < PaddedReferenceData::size; ++i) {
    BOOST_CHECK(local_ref.views_[i].image_dims_[0] ==
                local_ref.views_[i - 1].image_dims_[0]);
    BOOST_CHECK(local_ref.views_[i].image_dims_[1] ==
                local_ref.views_[i - 1].image_dims_[1]);
    BOOST_CHECK(local_ref.views_[i].image_dims_[2] ==
                local_ref.views_[i - 1].image_dims_[2]);
  }

  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);

  BOOST_CHECK_EQUAL_COLLECTIONS(
      local_ref.views_[0].image_dims_.begin(),
      local_ref.views_[0].image_dims_.end(),
      local_guesses.padded_psi(0, shape_to_padd_with)->shape(),
      local_guesses.padded_psi(0, shape_to_padd_with)->shape() + 3);
}

BOOST_AUTO_TEST_CASE(check_1st_two_iterations) {
  // setup
  PaddedReferenceData local_ref(reference);
  first_2_iterations local_guesses(local_guesses_of_2);

  // padd the psi to the same shape as the input images
  std::vector<int> shape_to_padd_with;
  local_ref.min_kernel_shape(shape_to_padd_with);

  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_,
                 local_guesses.minValue_);
  input.num_iterations_ = 1;
  image_stack input_psi = *local_guesses.padded_psi(0, shape_to_padd_with);

  inplace_cpu_deconvolve(input_psi.data(), input, 1);

  // check norms
  image_stack* padded_psi_1 = local_guesses.padded_psi(1, shape_to_padd_with);
  float l2norm = multiviewnative::l2norm(input_psi.data(), padded_psi_1->data(),
                                         input_psi.num_elements());
  BOOST_CHECK_LT(l2norm, 40);

  const float bottom_ratio = .35;
  const float upper_ratio = 1 - bottom_ratio;
  l2norm = multiviewnative::l2norm_within_limits(input_psi, *padded_psi_1,
                                                 bottom_ratio, upper_ratio);
  std::cout << "central norms: [" << bottom_ratio << "e," << upper_ratio
            << "e]**3\n"
            << "1-iter l2norm within limits \t" << l2norm << "\n";
  BOOST_REQUIRE_LT(l2norm, 1e-2);

  input_psi = *local_guesses.padded_psi(0, shape_to_padd_with);
  input.num_iterations_ = 2;
  inplace_cpu_deconvolve(input_psi.data(), input, 1);
  image_stack* padded_psi_2 = local_guesses.padded_psi(2, shape_to_padd_with);
  l2norm = multiviewnative::l2norm(input_psi.data(), padded_psi_2->data(),
                                   input_psi.num_elements());
  BOOST_CHECK_LT(l2norm, 70);
  l2norm = multiviewnative::l2norm_within_limits(input_psi, *padded_psi_2,
                                                 bottom_ratio, upper_ratio);
  std::cout << "central norms: [" << bottom_ratio << "e," << upper_ratio
            << "e]**3\n"
            << "2-iter l2norm within limits \t" << l2norm << "\n";
  BOOST_REQUIRE_LT(l2norm, 1e-2);
  // tear-down
  delete[] input.data_;
}

BOOST_AUTO_TEST_SUITE_END()

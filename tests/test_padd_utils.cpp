#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE PLAN_STORE
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iterator>

#include "padd_utils.h"
#include "test_fixtures.hpp"
#include "image_stack_utils.h"

typedef multiviewnative::zero_padd<multiviewnative::image_stack>
    wrap_around_padding;
typedef multiviewnative::no_padd<multiviewnative::image_stack> no_padding;

using namespace multiviewnative;

BOOST_FIXTURE_TEST_SUITE(no_padd, multiviewnative::default_3D_fixture)
BOOST_AUTO_TEST_CASE(constructs) {

  no_padding local(&image_dims_[0], &kernel_dims_[0]);

  for (unsigned i = 0; i < image_dims_.size(); ++i)
    BOOST_CHECK(image_dims_[i] == (int)local.extents_[i]);

  for (unsigned i = 0; i < image_dims_.size(); ++i)
    BOOST_CHECK(local.offsets_[i] == 0);
}

BOOST_AUTO_TEST_CASE(inserting_does_not_change_anything) {

  no_padding local(&image_dims_[0], &kernel_dims_[0]);
  std::fill(padded_image_.data(),
            padded_image_.data() + padded_image_.num_elements(), 0);
  local.insert_at_offsets(padded_one_, padded_image_);

  BOOST_CHECK(padded_one_ == padded_image_);
}

BOOST_AUTO_TEST_CASE(extents_match) {
  no_padding local(&padded_image_dims_[0], &kernel_dims_[0]);
  wrap_around_padding wlocal(&image_dims_[0], &kernel_dims_[0]);

  BOOST_CHECK_EQUAL_COLLECTIONS(local.extents_.begin(), local.extents_.end(),
                                wlocal.extents_.begin(), wlocal.extents_.end());
}

BOOST_AUTO_TEST_CASE(wrapped_inserting_horizontal) {

  no_padding local(&padded_image_dims_[0], &kernel_dims_[0]);
  wrap_around_padding wlocal(&image_dims_[0], &kernel_dims_[0]);

  std::fill(padded_image_.data(),
            padded_image_.data() + padded_image_.num_elements(), 0);
  image_stack no_padd_result = padded_image_;
  image_stack wrapped_padd_result = padded_image_;

  wlocal.wrapped_insert_at_offsets(horizont_kernel_, wrapped_padd_result);

  local.wrapped_insert_at_offsets(horizont_kernel_, no_padd_result);

  try {
    BOOST_REQUIRE(no_padd_result == wrapped_padd_result);
  }
  catch (...) {

    std::cout << "horizontal kernel:\n" << horizont_kernel_ << "\n\n"
              << "expected:\n" << wrapped_padd_result << "\n\n"
              << "received:\n" << no_padd_result << "\n";
  }

  std::fill(padded_image_.data(),
            padded_image_.data() + padded_image_.num_elements(), 0);
  no_padd_result = padded_image_;
  wrapped_padd_result = padded_image_;

  wlocal.wrapped_insert_at_offsets(vertical_kernel_, wrapped_padd_result);

  local.wrapped_insert_at_offsets(vertical_kernel_, no_padd_result);

  try {
    BOOST_REQUIRE(no_padd_result == wrapped_padd_result);
  }
  catch (...) {

    std::cout << "vertical kernel:\n" << vertical_kernel_ << "\n\n"
              << "expected:\n" << wrapped_padd_result << "\n\n"
              << "received:\n" << no_padd_result << "\n";
  }

  std::fill(padded_image_.data(),
            padded_image_.data() + padded_image_.num_elements(), 0);
  no_padd_result = padded_image_;
  wrapped_padd_result = padded_image_;

  wlocal.wrapped_insert_at_offsets(depth_kernel_, wrapped_padd_result);

  local.wrapped_insert_at_offsets(depth_kernel_, no_padd_result);

  try {
    BOOST_REQUIRE(no_padd_result == wrapped_padd_result);
  }
  catch (...) {

    std::cout << "kernel:\n" << depth_kernel_ << "\n\n"
              << "expected:\n" << wrapped_padd_result << "\n\n"
              << "received:\n" << no_padd_result << "\n";
  }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(wrap_padd, multiviewnative::default_3D_fixture)
BOOST_AUTO_TEST_CASE(constructs) {

  wrap_around_padding local(&image_dims_[0], &kernel_dims_[0]);

  for (unsigned i = 0; i < image_dims_.size(); ++i) {
    BOOST_CHECK(image_dims_[i] != (int)local.extents_[i]);
    BOOST_CHECK((image_dims_[i] + 2 * (kernel_dims_[i] / 2)) ==
                (int)local.extents_[i]);
  }

  for (unsigned i = 0; i < image_dims_.size(); ++i) {
    BOOST_CHECK(local.offsets_[i] != 0);
    BOOST_CHECK(int(local.offsets_[i]) == kernel_dims_[i] / 2);
  }
}

BOOST_AUTO_TEST_CASE(changes_image_when_inserting) {

  image_stack expected = padded_image_;
  std::fill(padded_image_.data(),
            padded_image_.data() + padded_image_.num_elements(), 0);

  wrap_around_padding local(&image_dims_[0], &kernel_dims_[0]);
  local.insert_at_offsets(image_, padded_image_);

  BOOST_CHECK(expected == padded_image_);

  BOOST_CHECK_EQUAL_COLLECTIONS(
      expected.data(), expected.data() + expected.num_elements(),
      padded_image_.data(),
      padded_image_.data() + padded_image_.num_elements());
}

BOOST_AUTO_TEST_CASE(changes_image_when_inserting_wrapped) {

  image_stack expected = padded_image_;
  std::fill(padded_image_.data(),
            padded_image_.data() + padded_image_.num_elements(), 0);

  wrap_around_padding local(&image_dims_[0], &kernel_dims_[0]);
  local.wrapped_insert_at_offsets(image_, padded_image_);

  BOOST_CHECK(expected != padded_image_);
}
BOOST_AUTO_TEST_SUITE_END()

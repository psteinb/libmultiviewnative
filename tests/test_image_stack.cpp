#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE INDEPENDENT_IMAGE_STACK
#include "boost/test/unit_test.hpp"
#include "image_stack_utils.h"
#include "test_fixtures.hpp"
#include <algorithm>

typedef multiviewnative::convolutionFixture3D<5, 9>
    more_then_default_3D_fixture;

BOOST_FIXTURE_TEST_SUITE(access_test_suite, multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(smaller_dims) {
  more_then_default_3D_fixture other;

  BOOST_CHECK_EQUAL(
      std::lexicographical_compare(
          this->one_.shape(),
          this->one_.shape() + multiviewnative::image_stack::dimensionality,
          other.one_.shape(),
          other.one_.shape() + multiviewnative::image_stack::dimensionality),
      true);
}

BOOST_AUTO_TEST_SUITE_END()

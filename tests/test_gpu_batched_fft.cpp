#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_BATCHED_FFTs
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>

#include "padd_utils.h"


  using namespace multiviewnative;

BOOST_FIXTURE_TEST_SUITE(batched_fft,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(try_two) {


  std::vector<image_stack> input(2,image_);

  BOOST_FAIL("not implemented yet!");

}


BOOST_AUTO_TEST_SUITE_END()

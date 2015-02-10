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
#include "test_fixtures.h"

typedef multiviewnative::zero_padd<multiviewnative::image_stack> wrap_around_padding;
typedef multiviewnative::no_padd<multiviewnative::image_stack> no_padding;

BOOST_FIXTURE_TEST_SUITE( no_padd , multiviewnative::default_3D_fixture )
BOOST_AUTO_TEST_CASE( default_constructs  )
{

BOOST_FAIL();
  
}
BOOST_AUTO_TEST_SUITE_END()

#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE INDEPENDENT_SHAPE_POINT
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>

#include "point.h"


BOOST_FIXTURE_TEST_SUITE( point_constructs , multiviewnative::default_3D_fixture )
BOOST_AUTO_TEST_CASE( constructs  )
{

  multiviewnative::shape_t any(3,128);
  BOOST_CHECK_EQUAL(any[0],128);
  BOOST_CHECK((any[0] == any[1]) && (any[2] == any[1]));
  BOOST_CHECK_THROW(any.at(3),std::exception);
  
}

BOOST_AUTO_TEST_CASE( smaller_than  )
{

  multiviewnative::shape_t smaller(3,128);
  multiviewnative::shape_t bigger(3,256);
  bool res = smaller < bigger;
  BOOST_CHECK(res);


}

BOOST_AUTO_TEST_CASE( smaller_than_asymmetric  )
{

  multiviewnative::shape_t smaller(3,128);
smaller[1] = 111;
smaller[2] = 255;

  multiviewnative::shape_t bigger(3,256);
  bool res = smaller < bigger;
  BOOST_CHECK(res);

}

BOOST_AUTO_TEST_CASE( equality  )
{

  multiviewnative::shape_t smaller(3,128);
  multiviewnative::shape_t bigger(3,256);
  bool res = !(smaller == bigger);
  BOOST_CHECK(res);
  BOOST_CHECK(smaller == smaller);

}
BOOST_AUTO_TEST_SUITE_END()

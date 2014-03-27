#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"

 
BOOST_FIXTURE_TEST_SUITE( access_test_suite, multiviewnative::default_3D_fixture )
   

BOOST_AUTO_TEST_CASE( first_value )
{
  
  BOOST_CHECK(image_[0][0][0] == 42.f);
}

BOOST_AUTO_TEST_CASE( center_value )
{
  
  BOOST_CHECK(image_[0][0][0] == 42.f);//?
}

// BOOST_AUTO_TEST_CASE( axis_length )
// {
  
//   BOOST_CHECK(image_.shape()[0] == image_.shape()[1] == image_.shape()[2] == 16 );
// }

BOOST_AUTO_TEST_SUITE_END()

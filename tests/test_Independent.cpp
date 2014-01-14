#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
#include "boost/test/unit_test.hpp"


class Independent_fixture
{
public:

  size_t evenDim;

  Independent_fixture():
    evenDim(42)
  {
    BOOST_MESSAGE( "setup fixture" );
  }
  
  virtual ~Independent_fixture()  { BOOST_MESSAGE( "teardown fixture" ); };
   
};
 
BOOST_FIXTURE_TEST_SUITE( Independent_suite, Independent_fixture )
   
BOOST_AUTO_TEST_CASE( fail )
{
  
  BOOST_CHECK(evenDim == 43);
}

BOOST_AUTO_TEST_CASE( success )
{
  
  BOOST_CHECK(evenDim == 42);
}

BOOST_AUTO_TEST_SUITE_END()

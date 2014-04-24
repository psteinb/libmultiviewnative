#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_DECONVOLVE
#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"

using namespace multiviewnative;

static const ReferenceData reference;

BOOST_AUTO_TEST_SUITE( deconvolve_psi0  )
   
BOOST_AUTO_TEST_CASE( reconstruct_anything )
{
  ReferenceData local(reference);
  
  BOOST_CHECK(local.views_[0].image_.num_elements()>0);
  BOOST_CHECK(local.views_[1].image_.num_elements()>0);
  BOOST_CHECK(local.views_[2].image_.num_elements()>0);
  BOOST_CHECK(local.views_[3].image_.num_elements()>0);
  BOOST_CHECK(local.views_[4].image_.num_elements()>0);
  BOOST_CHECK(local.views_[5].image_.num_elements()>0);
}

BOOST_AUTO_TEST_SUITE_END()

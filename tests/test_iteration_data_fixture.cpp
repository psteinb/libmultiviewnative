#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE INDEPENDENT_TIFF_PLAYGROUND
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include <numeric>

namespace fs = boost::filesystem;



BOOST_AUTO_TEST_SUITE(iteration_suite)

BOOST_AUTO_TEST_CASE(psi_paths_not_empty) {

  for (unsigned i = 0; i < 10; i++)
    BOOST_CHECK_MESSAGE(multiviewnative::all_iterations::instance().path(i)->empty() != true,
                        "psi " << i << " not loaded");
}

BOOST_AUTO_TEST_CASE(psi_stacks_not_empty) {

  for (unsigned i = 0; i < 10; i++)
    BOOST_CHECK(multiviewnative::all_iterations::instance().psi(i) != nullptr);
}

BOOST_AUTO_TEST_CASE(psi_stacks_nonzero_elements) {


  for (unsigned i = 0; i < 10; i++)
    BOOST_CHECK_GT(multiviewnative::all_iterations::instance().psi(i)->num_elements(), 0u);
}
BOOST_AUTO_TEST_SUITE_END()

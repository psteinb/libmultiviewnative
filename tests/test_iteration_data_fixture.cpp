#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE INDEPENDENT_TIFF_PLAYGROUND
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include <numeric>

namespace fs = boost::filesystem;

static const multiviewnative::all_iterations psis;

BOOST_AUTO_TEST_SUITE(iteration_suite)

BOOST_AUTO_TEST_CASE(psi_paths_not_empty) {
  multiviewnative::all_iterations local_psis(psis);

  for (unsigned i = 0; i < 10; i++)
    BOOST_CHECK_MESSAGE(local_psis.psi_paths_[i].empty() != true,
                        "psi " << i << " not loaded");
}

BOOST_AUTO_TEST_CASE(psi_stacks_not_empty) {
  multiviewnative::all_iterations local_psis(psis);

  for (unsigned i = 0; i < 10; i++)
    BOOST_CHECK_EQUAL(local_psis.psi(i)->num_elements(),
                      psis.psi(i)->num_elements());
}

BOOST_AUTO_TEST_CASE(psi_stacks_nonzero_elements) {
  multiviewnative::all_iterations local_psis(psis);

  for (unsigned i = 0; i < 10; i++)
    BOOST_CHECK_GT(local_psis.psi(i)->num_elements(), 0u);
}
BOOST_AUTO_TEST_SUITE_END()

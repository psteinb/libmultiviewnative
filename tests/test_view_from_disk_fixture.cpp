#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE INDEPENDENT_TIFF_PLAYGROUND
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include <numeric>

namespace fs = boost::filesystem;

BOOST_FIXTURE_TEST_SUITE(correctly_loaded_from_disk,
                         multiviewnative::ViewFromDisk)

BOOST_AUTO_TEST_CASE(path_and_files_exists_images_have_right_extents) {
  load(0);

  BOOST_CHECK(fs::is_directory(multiviewnative::path_to_test_images));
  BOOST_CHECK_MESSAGE(fs::is_regular_file(image_path_.str()),
                      " file at " << image_path_.str() << " does not exist\n");
  BOOST_CHECK_MESSAGE(fs::is_regular_file(kernel1_path_.str()),
                      " file at " << kernel1_path_.str()
                                  << " does not exist\n");
  BOOST_CHECK_MESSAGE(fs::is_regular_file(kernel2_path_.str()),
                      " file at " << kernel2_path_.str()
                                  << " does not exist\n");
  BOOST_CHECK_MESSAGE(fs::is_regular_file(weights_path_.str()),
                      " file at " << weights_path_.str()
                                  << " does not exist\n");

  BOOST_CHECK_EQUAL(image()->shape()[0], 252u);
  BOOST_CHECK_EQUAL(image()->shape()[1], 212u);
  BOOST_CHECK_EQUAL(image()->shape()[2], 220u);

  BOOST_CHECK_EQUAL(kernel1()->shape()[0], 19u);
  BOOST_CHECK_EQUAL(kernel1()->shape()[1], 19u);
  BOOST_CHECK_EQUAL(kernel1()->shape()[2], 85u);

  BOOST_CHECK_EQUAL(kernel2()->shape()[0], 19u);
  BOOST_CHECK_EQUAL(kernel2()->shape()[1], 19u);
  BOOST_CHECK_EQUAL(kernel2()->shape()[2], 85u);

  BOOST_CHECK_EQUAL(weights()->shape()[0], 252u);
  BOOST_CHECK_EQUAL(weights()->shape()[1], 212u);
  BOOST_CHECK_EQUAL(weights()->shape()[2], 220u);
}

BOOST_AUTO_TEST_SUITE_END()

// static const multiviewnative::ViewFromDisk view0(0);

// BOOST_FIXTURE_TEST_SUITE( views_suite, multiviewnative::ViewFromDisk )

// BOOST_AUTO_TEST_CASE( images_have_right_extents )
// {
//   load(view0);

//   BOOST_CHECK_EQUAL(image()->shape()[0], 252);
//   BOOST_CHECK_EQUAL(image()->shape()[1], 212);
//   BOOST_CHECK_EQUAL(image()->shape()[2], 220);

//   BOOST_CHECK_EQUAL(kernel1()->shape()[0], 19);
//   BOOST_CHECK_EQUAL(kernel1()->shape()[1], 19);
//   BOOST_CHECK_EQUAL(kernel1()->shape()[2], 85);

//   BOOST_CHECK_EQUAL(kernel2()->shape()[0], 19);
//   BOOST_CHECK_EQUAL(kernel2()->shape()[1], 19);
//   BOOST_CHECK_EQUAL(kernel2()->shape()[2], 85);

//   BOOST_CHECK_EQUAL(weights()->shape()[0], 252);
//   BOOST_CHECK_EQUAL(weights()->shape()[1], 212);
//   BOOST_CHECK_EQUAL(weights()->shape()[2], 220);

// }

// BOOST_AUTO_TEST_SUITE_END()

// static const multiviewnative::IterationData psis;

// BOOST_AUTO_TEST_SUITE( iteration_suite )

// BOOST_AUTO_TEST_CASE( psi_paths_not_empty )
// {
//   multiviewnative::IterationData local_psis(psis);

//   for(int i = 0;i<10;i++)
//     BOOST_CHECK_MESSAGE(local_psis.psi_paths_[i].empty()!=true, "psi " << i
// << " not loaded");

// }

// BOOST_AUTO_TEST_CASE( psi_stacks_not_empty )
// {
//   multiviewnative::IterationData local_psis(psis);

//   for(int i = 0;i<10;i++)
//     BOOST_CHECK_EQUAL(local_psis.psi(i)->num_elements(),psis.psi(i)->num_elements());

// }

// BOOST_AUTO_TEST_SUITE_END()

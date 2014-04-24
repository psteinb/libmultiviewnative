#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE INDEPENDENT_TIFF_PLAYGROUND
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include <numeric>

namespace fs = boost::filesystem;



BOOST_FIXTURE_TEST_SUITE( correctly_loaded_from_disk, multiviewnative::ViewFromDisk )
   
BOOST_AUTO_TEST_CASE( path_and_files_exists_images_have_right_extents )
{
  load(0);
  
  BOOST_CHECK(fs::is_directory(multiviewnative::path_to_test_images));
  BOOST_CHECK_MESSAGE(fs::is_regular_file(image_path_  .str()), " file at " << image_path_  .str()<< " does not exist\n");
  BOOST_CHECK_MESSAGE(fs::is_regular_file(kernel1_path_.str()), " file at " << kernel1_path_.str()<< " does not exist\n");
  BOOST_CHECK_MESSAGE(fs::is_regular_file(kernel2_path_.str()), " file at " << kernel2_path_.str()<< " does not exist\n");
  BOOST_CHECK_MESSAGE(fs::is_regular_file(weights_path_.str()), " file at " << weights_path_.str()<< " does not exist\n");

  BOOST_CHECK_EQUAL(image_.shape()[0], 252);
  BOOST_CHECK_EQUAL(image_.shape()[1], 212);
  BOOST_CHECK_EQUAL(image_.shape()[2], 220);

  BOOST_CHECK_EQUAL(kernel1_.shape()[0], 19);
  BOOST_CHECK_EQUAL(kernel1_.shape()[1], 19);
  BOOST_CHECK_EQUAL(kernel1_.shape()[2], 85);

  BOOST_CHECK_EQUAL(kernel2_.shape()[0], 19);
  BOOST_CHECK_EQUAL(kernel2_.shape()[1], 19);
  BOOST_CHECK_EQUAL(kernel2_.shape()[2], 85);  

  BOOST_CHECK_EQUAL(weights_.shape()[0], 252);
  BOOST_CHECK_EQUAL(weights_.shape()[1], 212);
  BOOST_CHECK_EQUAL(weights_.shape()[2], 220);

}

BOOST_AUTO_TEST_SUITE_END()

static const multiviewnative::ViewFromDisk view0(0);

BOOST_FIXTURE_TEST_SUITE( fixture_complete, multiviewnative::ViewFromDisk )
   
BOOST_AUTO_TEST_CASE( images_have_right_extents )
{
  load(view0);

  BOOST_CHECK_EQUAL(image_.shape()[0], 252);
  BOOST_CHECK_EQUAL(image_.shape()[1], 212);
  BOOST_CHECK_EQUAL(image_.shape()[2], 220);

  BOOST_CHECK_EQUAL(kernel1_.shape()[0], 19);
  BOOST_CHECK_EQUAL(kernel1_.shape()[1], 19);
  BOOST_CHECK_EQUAL(kernel1_.shape()[2], 85);

  BOOST_CHECK_EQUAL(kernel2_.shape()[0], 19);
  BOOST_CHECK_EQUAL(kernel2_.shape()[1], 19);
  BOOST_CHECK_EQUAL(kernel2_.shape()[2], 85);  

  BOOST_CHECK_EQUAL(weights_.shape()[0], 252);
  BOOST_CHECK_EQUAL(weights_.shape()[1], 212);
  BOOST_CHECK_EQUAL(weights_.shape()[2], 220);

}

BOOST_AUTO_TEST_SUITE_END()

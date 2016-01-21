#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE INDEPENDENT_TIFF_PLAYGROUND
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include "tiff_fixtures_helpers.hpp"
#include <numeric>

namespace fs = boost::filesystem;
namespace mvn = multiviewnative;

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

  std::vector<int> kernel1_shape = mvn::shape_of<mvn::kernel1>(0);
  std::vector<int> kernel2_shape = mvn::shape_of<mvn::kernel2>(0);

  for(int i = 0;i<3;++i){
    BOOST_CHECK_EQUAL(kernel1()->shape()[i], kernel1_shape[i]);
    BOOST_CHECK_EQUAL(kernel2()->shape()[i], kernel2_shape[i]);
  }
  
}

BOOST_AUTO_TEST_CASE(adaptive_path_exists) {
  
  fs::path image_view_path = multiviewnative::path_to_test_images;
  image_view_path /= "input_view_0.tif";
  BOOST_CHECK_EQUAL(fs::exists(image_view_path),true);

  BOOST_CHECK_EQUAL(fs::exists(mvn::path_to<mvn::input  >(0)),true);
  BOOST_CHECK_EQUAL(fs::exists(mvn::path_to<mvn::kernel1>(0)),true);
  BOOST_CHECK_EQUAL(fs::exists(mvn::path_to<mvn::kernel2>(0)),true);
  BOOST_CHECK_EQUAL(fs::exists(mvn::path_to<mvn::weights>(0)),true);

  std::vector<int> kernel1_shape = mvn::shape_of<mvn::kernel1>(2);
  BOOST_CHECK_GT(kernel1_shape.size(),0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(refernce_iterations)

BOOST_AUTO_TEST_CASE(instance) {
  BOOST_CHECK(mvn::first_2_iterations::instance().lambda()>0);
}

BOOST_AUTO_TEST_SUITE_END()

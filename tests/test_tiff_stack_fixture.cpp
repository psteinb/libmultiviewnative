#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE INDEPENDENT_TIFF_PLAYGROUND

#include <numeric>
#include <string>

#include "boost/test/unit_test.hpp"
#include "boost/filesystem.hpp"

#include "tiff_fixtures.hpp"
#include "tiff_fixtures_helpers.hpp"
#include "test_algorithms.hpp"
#include "tests_config.h"


namespace fs = boost::filesystem;
namespace mvn = multiviewnative;


const static multiviewnative::tiff_stack image_view_1(
    // "/scratch/steinbac/multiview_data/input_view_0.tif"
						      mvn::path_to<mvn::input>(0).string() 
						      );

const static multiviewnative::tiff_stack kernel1_view_2(
    // "/scratch/steinbac/multiview_data/input_view_0.tif"
						      mvn::path_to<mvn::kernel1>(2).string() 
						      );


BOOST_FIXTURE_TEST_SUITE(tiff_loaded, multiviewnative::tiff_stack)

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


BOOST_AUTO_TEST_CASE(path_and_files_exists_images_have_right_extents) {

  fs::path location = mvn::path_to<mvn::kernel1>(2);
  std::string path = location.string();

  if (!fs::exists(location)) {
    BOOST_FAIL("test file does not exist");
  }
  load(location.string());

  BOOST_CHECK_MESSAGE(!empty(), "image under " << location
                                               << " was not loaded");
  BOOST_CHECK_EQUAL(location.string().size(), stack_path_.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(path.begin(), path.end(), stack_path_.begin(),
                                stack_path_.end());

  std::vector<int> kernel1_shape = mvn::shape_of<mvn::kernel1>(2);

  if(kernel1_shape.empty()){
    BOOST_CHECK_EQUAL(stack_.shape()[2], 23u);
    BOOST_CHECK_EQUAL(stack_.shape()[1], 81u);
    BOOST_CHECK_EQUAL(stack_.shape()[0], 57u);
  } else {

    for( unsigned int i = 0;i<kernel1_shape.size();++i)
      BOOST_CHECK_EQUAL(stack_.shape()[i], kernel1_shape[i]);
        
  }
}

BOOST_AUTO_TEST_CASE(copy_constructor) {
  multiviewnative::tiff_stack local(kernel1_view_2);

  BOOST_CHECK_MESSAGE(!local.empty(), "image under " << kernel1_view_2.stack_path_
		      << " was not copied-in");
  BOOST_CHECK_EQUAL(local.stack_path_.size(), kernel1_view_2.stack_path_.size());

  std::vector<int> kernel1_shape = mvn::shape_of<mvn::kernel1>(2);
  if(kernel1_shape.empty()){
    BOOST_CHECK_EQUAL(local.stack_.shape()[2], 23u);
    BOOST_CHECK_EQUAL(local.stack_.shape()[1], 81u);
    BOOST_CHECK_EQUAL(local.stack_.shape()[0], 57u);
  }
  else {

    for( unsigned int i = 0;i<kernel1_shape.size();++i)
      BOOST_CHECK_EQUAL(local.stack_.shape()[i], kernel1_shape[i]);
        
  }
  
}

BOOST_AUTO_TEST_CASE(assign_constructor) {
  multiviewnative::tiff_stack local = kernel1_view_2;

  BOOST_CHECK_MESSAGE(!local.empty(), "image under " << kernel1_view_2.stack_path_
		      << " was not assinged-in");
  BOOST_CHECK_EQUAL(local.stack_path_.size(), kernel1_view_2.stack_path_.size());
  std::vector<int> kernel1_shape = mvn::shape_of<mvn::kernel1>(2);
  if(kernel1_shape.empty()){
    BOOST_CHECK_EQUAL(local.stack_.shape()[2], 23u);
    BOOST_CHECK_EQUAL(local.stack_.shape()[1], 81u);
    BOOST_CHECK_EQUAL(local.stack_.shape()[0], 57u);
  }
  else {

    for( unsigned int i = 0;i<kernel1_shape.size();++i)
      BOOST_CHECK_EQUAL(local.stack_.shape()[i], kernel1_shape[i]);
        
  }

}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(simple_write_to_disk)

BOOST_AUTO_TEST_CASE(write_234_image) {
  multiviewnative::image_stack test(boost::extents[4][3][2]);
  for (uint32_t i = 0; i < test.num_elements(); ++i) test.data()[i] = i;

  multiviewnative::write_image_stack(test, "./test.tif");
  multiviewnative::tiff_stack reloaded("./test.tif");

  BOOST_CHECK_EQUAL(test.num_elements(), reloaded.stack_.num_elements());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      test.data(), test.data() + test.num_elements(), reloaded.stack_.data(),
      reloaded.stack_.data() + reloaded.stack_.num_elements());
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(write_to_disk, multiviewnative::tiff_stack)

BOOST_AUTO_TEST_CASE(write_reload_kernel1_view_2) {
  fs::path loc = multiviewnative::path_to_test_images;
  loc += "input_view_1.tif";

  if (!fs::exists(loc)) {
    BOOST_FAIL("unable to load " << loc.string());
  }
  load(loc.string());

  multiviewnative::write_image_stack(stack_, "./test.tif");

  multiviewnative::tiff_stack reloaded("./test.tif");

  BOOST_CHECK_EQUAL(stack_.num_elements(), reloaded.stack_.num_elements());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      stack_.data(), stack_.data() + stack_.num_elements(),
      reloaded.stack_.data(),
      reloaded.stack_.data() + reloaded.stack_.num_elements());

  float l2norm = multiviewnative::l2norm(stack_.data(), reloaded.stack_.data(),
                                         stack_.num_elements());
  BOOST_CHECK_CLOSE(l2norm, 0.f, 1e-3);
}

BOOST_AUTO_TEST_CASE(write_reload_kernel1) {
  fs::path loc = multiviewnative::path_to_test_images;
  loc += "kernel1_view_1.tif";

  if (!fs::exists(loc)) {
    BOOST_FAIL("unable to load " << loc.string());
  }
  load(loc.string());

  multiviewnative::write_image_stack(stack_, "./test.tif");

  multiviewnative::tiff_stack reloaded("./test.tif");

  BOOST_CHECK_EQUAL(stack_.num_elements(), reloaded.stack_.num_elements());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      stack_.data(), stack_.data() + stack_.num_elements(),
      reloaded.stack_.data(),
      reloaded.stack_.data() + reloaded.stack_.num_elements());

  float l2norm = multiviewnative::l2norm(stack_.data(), reloaded.stack_.data(),
                                         stack_.num_elements());
  BOOST_CHECK_CLOSE(l2norm, 0.f, 1e-3);
}

BOOST_AUTO_TEST_SUITE_END()

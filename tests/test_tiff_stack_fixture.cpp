#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE INDEPENDENT_TIFF_PLAYGROUND
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include "test_algorithms.hpp"
#include "tests_config.h"
#include <numeric>

namespace fs = boost::filesystem;

const static multiviewnative::tiff_stack image_view_1(
    "/scratch/steinbac/multiview_data/input_view_0.tif");

BOOST_FIXTURE_TEST_SUITE(tiff_loaded, multiviewnative::tiff_stack)

BOOST_AUTO_TEST_CASE(path_and_files_exists_images_have_right_extents) {
  fs::path location = "/scratch/steinbac/multiview_data/input_view_0.tif";
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
  BOOST_CHECK_EQUAL(stack_.shape()[2], 252u);
  BOOST_CHECK_EQUAL(stack_.shape()[1], 212u);
  BOOST_CHECK_EQUAL(stack_.shape()[0], 220u);
}

BOOST_AUTO_TEST_CASE(copy_constructor) {
  multiviewnative::tiff_stack local(image_view_1);

  BOOST_CHECK_MESSAGE(!local.empty(), "image under " << image_view_1.stack_path_
                                                     << " was not copied-in");
  BOOST_CHECK_EQUAL(local.stack_path_.size(), image_view_1.stack_path_.size());

  BOOST_CHECK_EQUAL(local.stack_.shape()[2], 252u);
  BOOST_CHECK_EQUAL(local.stack_.shape()[1], 212u);
  BOOST_CHECK_EQUAL(local.stack_.shape()[0], 220u);
}

BOOST_AUTO_TEST_CASE(assign_constructor) {
  multiviewnative::tiff_stack local = image_view_1;

  BOOST_CHECK_MESSAGE(!local.empty(), "image under " << image_view_1.stack_path_
                                                     << " was not assinged-in");
  BOOST_CHECK_EQUAL(local.stack_path_.size(), image_view_1.stack_path_.size());

  BOOST_CHECK_EQUAL(local.stack_.shape()[2], 252u);
  BOOST_CHECK_EQUAL(local.stack_.shape()[1], 212u);
  BOOST_CHECK_EQUAL(local.stack_.shape()[0], 220u);
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

BOOST_AUTO_TEST_CASE(write_reload_image_view_1) {
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

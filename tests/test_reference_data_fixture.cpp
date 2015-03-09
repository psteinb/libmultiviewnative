#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TEST_REFERENCE_DATA_FIXTURE
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include "convert_tiff_fixtures.hpp"
#include <numeric>
#include "test_algorithms.hpp"

namespace fs = boost::filesystem;

static const multiviewnative::RawReferenceData ref;

BOOST_AUTO_TEST_SUITE(ref_suite)

BOOST_AUTO_TEST_CASE(ref_paths_not_empty) {
  multiviewnative::RawReferenceData local_ref(ref);

  for (int i = 0; i < 6; i++)
    BOOST_CHECK_MESSAGE(local_ref.views_[i].image_path_.str().empty() != true,
                        "ref view " << i << " not loaded");
}

BOOST_AUTO_TEST_CASE(ref_stacks_equal) {
  multiviewnative::RawReferenceData local_ref(ref);

  for (int i = 0; i < 6; i++)
    BOOST_CHECK_EQUAL(local_ref.views_[i].image()->num_elements(),
                      ref.views_[i].image()->num_elements());
}

BOOST_AUTO_TEST_CASE(ref_stacks_equal_views_files) {
  multiviewnative::RawReferenceData local_ref(ref);
  std::stringstream path("");
  for (int i = 0; i < 6; i++) {
    path.str("");
    path << multiviewnative::path_to_test_images << "image_view_" << i
         << ".tif";
    multiviewnative::tiff_stack current(path.str());

    BOOST_CHECK_CLOSE(
        multiviewnative::l2norm(current.stack_.data(),
                                local_ref.views_[i].image()->data(),
                                local_ref.views_[i].image()->num_elements()),
        0., 1e-2);
  }
}

BOOST_AUTO_TEST_CASE(ref_stacks_equal_kernel1_files) {
  multiviewnative::RawReferenceData local_ref(ref);
  std::stringstream path("");
  for (int i = 0; i < 6; i++) {
    path.str("");
    path << multiviewnative::path_to_test_images << "kernel1_view_" << i
         << ".tif";
    multiviewnative::tiff_stack current(path.str());

    BOOST_CHECK_CLOSE(
        multiviewnative::l2norm(current.stack_.data(),
                                local_ref.views_[i].kernel1()->data(),
                                local_ref.views_[i].kernel1()->num_elements()),
        0., 1e-2);
  }
}

BOOST_AUTO_TEST_CASE(ref_stacks_equal_kernel2_files) {
  multiviewnative::RawReferenceData local_ref(ref);
  std::stringstream path("");
  for (int i = 0; i < 6; i++) {
    path.str("");
    path << multiviewnative::path_to_test_images << "kernel2_view_" << i
         << ".tif";
    multiviewnative::tiff_stack current(path.str());

    BOOST_CHECK_CLOSE(
        multiviewnative::l2norm(current.stack_.data(),
                                local_ref.views_[i].kernel2()->data(),
                                local_ref.views_[i].kernel2()->num_elements()),
        0., 1e-2);
  }
}

BOOST_AUTO_TEST_CASE(ref_stacks_equal_weights_files) {
  multiviewnative::RawReferenceData local_ref(ref);
  std::stringstream path("");
  for (int i = 0; i < 6; i++) {
    path.str("");
    path << multiviewnative::path_to_test_images << "weights_view_" << i
         << ".tif";
    multiviewnative::tiff_stack current(path.str());

    BOOST_CHECK_CLOSE(
        multiviewnative::l2norm(current.stack_.data(),
                                local_ref.views_[i].weights()->data(),
                                local_ref.views_[i].weights()->num_elements()),
        0., 1e-2);
  }
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(convert_suite)
BOOST_AUTO_TEST_CASE(workspace_data_nonzero) {
  multiviewnative::RawReferenceData local_ref(ref);

  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, 0.006, 0.0001f);

  BOOST_CHECK_MESSAGE(input.data_, "input data was not created");

  for (int i = 0; i < 6; i++) {
    BOOST_CHECK_EQUAL(local_ref.views_[i].image()->num_elements(),
                      (unsigned)input.data_[i].image_dims_[0] *
                          input.data_[i].image_dims_[1] *
                          input.data_[i].image_dims_[2]);

    BOOST_CHECK_EQUAL_COLLECTIONS(local_ref.views_[i].image()->data(),
                                  local_ref.views_[i].image()->data() + 64,
                                  &input.data_[i].image_[0],
                                  &input.data_[i].image_[0] + 64);

    BOOST_CHECK_EQUAL_COLLECTIONS(
        local_ref.views_[i].image()->data() +
            local_ref.views_[i].image()->num_elements() - 64,
        local_ref.views_[i].image()->data() +
            local_ref.views_[i].image()->num_elements(),
        &input.data_[i].image_[0] +
            local_ref.views_[i].image()->num_elements() - 64,
        &input.data_[i].image_[0] +
            local_ref.views_[i].image()->num_elements());
  }

  delete[] input.data_;
}
BOOST_AUTO_TEST_SUITE_END()

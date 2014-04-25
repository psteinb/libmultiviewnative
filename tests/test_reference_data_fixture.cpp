#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE TEST_REFERENCE_DATA_FIXTURE
#include "boost/test/unit_test.hpp"
#include <boost/filesystem.hpp>
#include "tiff_fixtures.hpp"
#include "convert_tiff_fixtures.hpp"
#include <numeric>

namespace fs = boost::filesystem;

static const multiviewnative::ReferenceData ref;

BOOST_AUTO_TEST_SUITE( ref_suite )
   
BOOST_AUTO_TEST_CASE( ref_paths_not_empty )
{
  multiviewnative::ReferenceData local_ref(ref);

  for(int i = 0;i<6;i++)
    BOOST_CHECK_MESSAGE(local_ref.views_[i].image_path_.str().empty()!=true, "ref view " << i << " not loaded");

}

BOOST_AUTO_TEST_CASE( ref_stacks_equal )
{
  multiviewnative::ReferenceData local_ref(ref);

  for(int i = 0;i<6;i++)
    BOOST_CHECK_EQUAL(local_ref.views_[i].image()->num_elements(),ref.views_[i].image()->num_elements());

}


BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( convert_suite )
BOOST_AUTO_TEST_CASE( workspace_data_nonzero )
{
  multiviewnative::ReferenceData local_ref(ref);

  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input);

  BOOST_CHECK_MESSAGE(input.data_, "input data was not created");

  for(int i = 0;i<6;i++){
    BOOST_CHECK_EQUAL(local_ref.views_[i].image()->num_elements(), input.data_[i].image_dims_[0]*input.data_[i].image_dims_[1]*input.data_[i].image_dims_[2]);

    BOOST_CHECK_EQUAL_COLLECTIONS(local_ref.views_[i].image()->data(), local_ref.views_[i].image()->data()+64, 
				  &input.data_[i].image_[0], &input.data_[i].image_[0] + 64);

    BOOST_CHECK_EQUAL_COLLECTIONS(local_ref.views_[i].image()->data() + local_ref.views_[i].image()->num_elements() - 64, 
				  local_ref.views_[i].image()->data()+local_ref.views_[i].image()->num_elements(),  
				  &input.data_[i].image_[0]+local_ref.views_[i].image()->num_elements() - 64, 
				  &input.data_[i].image_[0]+local_ref.views_[i].image()->num_elements());
  }

  delete [] input.data_;

}
BOOST_AUTO_TEST_SUITE_END()

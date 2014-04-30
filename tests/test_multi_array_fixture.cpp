#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE INDEPENDENT_MULTIARRAY_PLAYGROUND
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>

 
BOOST_FIXTURE_TEST_SUITE( access_test_suite, multiviewnative::default_3D_fixture )
   

BOOST_AUTO_TEST_CASE( first_value )
{
  
  BOOST_CHECK_EQUAL(image_[0][0][0],0);
}

BOOST_AUTO_TEST_CASE( center_value )
{
  using namespace multiviewnative;
  int center_one_dim_size = image_dims_[0]/2;
  BOOST_CHECK_EQUAL(image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], 292);
}

BOOST_AUTO_TEST_CASE( axis_length )
{
  BOOST_CHECK(image_.shape()[0] == 8 && image_.shape()[1] == 8 && image_.shape()[2] == 8 );
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( kernel_setup, multiviewnative::default_3D_fixture )
   

BOOST_AUTO_TEST_CASE( identity_kernely )
{
  
  BOOST_CHECK_NE(identity_kernel_[1][1][1],0.f);
  float sum = std::accumulate(identity_kernel_.data(),identity_kernel_.data()+kernel_size_,0.);
  BOOST_CHECK_EQUAL(sum,1.f);
  
}

BOOST_AUTO_TEST_CASE( horizont_kernel )
{
  
  // printKernel(horizont_kernel_.data());
  BOOST_CHECK_EQUAL(horizont_kernel_[0][1][1],1.f);
  BOOST_CHECK_EQUAL(horizont_kernel_[1][1][1],2.f);
  BOOST_CHECK_EQUAL(horizont_kernel_[2][1][1],3.f);
  

}

BOOST_AUTO_TEST_CASE( vertical_kernel )
{
  
  // printKernel(vertical_kernel_.data());
  BOOST_CHECK_EQUAL(vertical_kernel_[1][0][1],1.f);
  BOOST_CHECK_EQUAL(vertical_kernel_[1][1][1],2.f);
  BOOST_CHECK_EQUAL(vertical_kernel_[1][2][1],3.f);
  
}

BOOST_AUTO_TEST_CASE( depth_kernel )
{
  // printKernel(depth_kernel_.data());
  BOOST_CHECK_EQUAL(depth_kernel_[1][1][0],1.f);
  BOOST_CHECK_EQUAL(depth_kernel_[1][1][1],2.f);
  BOOST_CHECK_EQUAL(depth_kernel_[1][1][2],3.f);
  
}


BOOST_AUTO_TEST_SUITE_END()


BOOST_FIXTURE_TEST_SUITE( convolution_works, multiviewnative::default_3D_fixture )
   

BOOST_AUTO_TEST_CASE( convolve_by_all1 )
{
  int center_one_dim_size = image_dims_[0]/2;
  BOOST_CHECK_NE(image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], image_folded_by_all1_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);

  float value = 0.f;
  for(int z_shift=-1;z_shift<=1;++z_shift){
    for(int y_shift=-1;y_shift<=1;++y_shift){
      for(int x_shift=-1;x_shift<=1;++x_shift){
	value += image_[center_one_dim_size+x_shift][center_one_dim_size+y_shift][center_one_dim_size+z_shift];
      }
    }
  }
  BOOST_CHECK_EQUAL(value, image_folded_by_all1_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);

}

BOOST_AUTO_TEST_CASE( convolve_by_horizontal )
{
  int center_one_dim_size = image_dims_[0]/2;
  
  BOOST_CHECK_NE(image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], image_folded_by_horizontal_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
  float intermediate = 2*image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
    + 3*image_[center_one_dim_size-1][center_one_dim_size][center_one_dim_size] 
    + 1*image_[center_one_dim_size+1][center_one_dim_size][center_one_dim_size];

  BOOST_CHECK_EQUAL(intermediate, image_folded_by_horizontal_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
    // based on bug from Apr 2, 2014
  BOOST_CHECK_CLOSE(image_folded_by_horizontal_[0][0][6],1153.f, 1e-4);
  BOOST_CHECK_CLOSE(image_folded_by_horizontal_[0][0][7],1345.f, 1e-4);


}

BOOST_AUTO_TEST_CASE( convolve_by_vertical )
{
  int center_one_dim_size = image_dims_[0]/2;
  
  BOOST_CHECK_NE(image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], image_folded_by_vertical_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
  float intermediate = 2*image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
    + 3*image_[center_one_dim_size][center_one_dim_size-1][center_one_dim_size] 
    + 1*image_[center_one_dim_size][center_one_dim_size+1][center_one_dim_size];

  BOOST_CHECK_EQUAL(intermediate, image_folded_by_vertical_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
}

BOOST_AUTO_TEST_CASE( convolve_by_depth )
{
  int center_one_dim_size = image_dims_[0]/2;
  
  BOOST_CHECK_NE(image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], image_folded_by_depth_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
  float intermediate = 2*image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
    + 3*image_[center_one_dim_size][center_one_dim_size][center_one_dim_size-1] 
    + 1*image_[center_one_dim_size][center_one_dim_size][center_one_dim_size+1];

  BOOST_CHECK_EQUAL(intermediate, image_folded_by_depth_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
}

BOOST_AUTO_TEST_CASE( convolve_by_asymm_cross_kernel )
{
  
  float sum_expected = std::accumulate(asymm_cross_kernel_.data(),asymm_cross_kernel_.data() + asymm_cross_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(one_folded_by_asymm_cross_kernel_.data(),one_folded_by_asymm_cross_kernel_.data() + one_folded_by_asymm_cross_kernel_.num_elements(),0.f);
  
  BOOST_CHECK_EQUAL(sum_expected,sum_received);
}

BOOST_AUTO_TEST_CASE( convolve_by_asymm_one_kernel )
{
  
  float sum_expected = std::accumulate(asymm_one_kernel_.data(),asymm_one_kernel_.data() + asymm_one_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(one_folded_by_asymm_one_kernel_.data(),one_folded_by_asymm_one_kernel_.data() + one_folded_by_asymm_one_kernel_.num_elements(),0.f);
  
  BOOST_CHECK_EQUAL(sum_expected,sum_received);
}

BOOST_AUTO_TEST_CASE( convolve_by_asymm_identity_kernel )
{
  
  float sum_expected = std::accumulate(asymm_identity_kernel_.data(),asymm_identity_kernel_.data() + asymm_identity_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(one_folded_by_asymm_identity_kernel_.data(),one_folded_by_asymm_identity_kernel_.data() + one_folded_by_asymm_identity_kernel_.num_elements(),0.f);
  
  BOOST_CHECK_EQUAL(sum_expected,sum_received);
}
BOOST_AUTO_TEST_SUITE_END()

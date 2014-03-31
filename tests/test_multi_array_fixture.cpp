#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>

 
BOOST_FIXTURE_TEST_SUITE( access_test_suite, multiviewnative::default_3D_fixture )
   

BOOST_AUTO_TEST_CASE( first_value )
{
  
  BOOST_CHECK(image_[0][0][0] == 42);
}

BOOST_AUTO_TEST_CASE( center_value )
{
  int center_one_dim_size = image_dims_[0]/2;
  BOOST_CHECK(image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] == 42);
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
  BOOST_CHECK_NE(padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], padded_image_folded_by_all1_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);

  float value = 0.f;
  for(int z_shift=-1;z_shift<=1;++z_shift){
    for(int y_shift=-1;y_shift<=1;++y_shift){
      for(int x_shift=-1;x_shift<=1;++x_shift){
	value += padded_image_[center_one_dim_size+x_shift][center_one_dim_size+y_shift][center_one_dim_size+z_shift];
      }
    }
  }
  BOOST_CHECK_EQUAL(value, padded_image_folded_by_all1_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);

}

BOOST_AUTO_TEST_CASE( convolve_by_horizontal )
{
  int center_one_dim_size = image_dims_[0]/2;
  
  BOOST_CHECK_NE(padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], padded_image_folded_by_horizontal_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
  float intermediate = 2*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
    + 3*padded_image_[center_one_dim_size-1][center_one_dim_size][center_one_dim_size] 
    + 1*padded_image_[center_one_dim_size+1][center_one_dim_size][center_one_dim_size];

  BOOST_CHECK_EQUAL(intermediate, padded_image_folded_by_horizontal_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
}

BOOST_AUTO_TEST_CASE( convolve_by_vertical )
{
  int center_one_dim_size = image_dims_[0]/2;
  
  BOOST_CHECK_NE(padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], padded_image_folded_by_vertical_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
  float intermediate = 2*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
    + 3*padded_image_[center_one_dim_size][center_one_dim_size-1][center_one_dim_size] 
    + 1*padded_image_[center_one_dim_size][center_one_dim_size+1][center_one_dim_size];

  BOOST_CHECK_EQUAL(intermediate, padded_image_folded_by_vertical_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
}

BOOST_AUTO_TEST_CASE( convolve_by_depth )
{
  int center_one_dim_size = image_dims_[0]/2;
  
  BOOST_CHECK_NE(padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], padded_image_folded_by_depth_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
  float intermediate = 2*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
    + 3*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size-1] 
    + 1*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size+1];

  BOOST_CHECK_EQUAL(intermediate, padded_image_folded_by_depth_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
}
BOOST_AUTO_TEST_SUITE_END()

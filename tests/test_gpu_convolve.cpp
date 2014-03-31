#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "multiviewnative.h"


BOOST_FIXTURE_TEST_SUITE( convolution_works, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( convolve_by_identity )
{
  
  float* image = new float[image_size_];

  std::copy(padded_image_.origin(), padded_image_.origin() + image_size_,image);

  convolution3DfftCUDAInPlace(image, &image_dims_[0], 
			      horizont_kernel_.data(),&kernel_dims_[0],
			      0);

  float * reference = image_.data();
  BOOST_CHECK_EQUAL_COLLECTIONS( image, image+image_size_/2, reference, reference + image_size_/2);
 
  delete [] image;
}


BOOST_AUTO_TEST_CASE( convolve_by_horizontal )
{
  
  float* image = new float[image_size_];

  std::copy(padded_image_.origin(), padded_image_.origin() + image_size_,image);

  convolution3DfftCUDAInPlace(image, &image_dims_[0], 
			      horizont_kernel_.data(),&kernel_dims_[0],
			      0);

  float * reference = padded_image_folded_by_horizontal_.data();
  BOOST_CHECK_EQUAL_COLLECTIONS( image, image+image_size_/2, reference, reference + image_size_/2);
 
  delete [] image;
}
// BOOST_AUTO_TEST_CASE( convolve_by_vertical )
// {
//   int center_one_dim_size = image_dims_[0]/2;
  
//   BOOST_CHECK_NE(padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], padded_image_folded_by_vertical_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
//   float intermediate = 2*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
//     + 3*padded_image_[center_one_dim_size][center_one_dim_size-1][center_one_dim_size] 
//     + 1*padded_image_[center_one_dim_size][center_one_dim_size+1][center_one_dim_size];

//   BOOST_CHECK_EQUAL(intermediate, padded_image_folded_by_vertical_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
// }

// BOOST_AUTO_TEST_CASE( convolve_by_depth )
// {
//   int center_one_dim_size = image_dims_[0]/2;
  
//   BOOST_CHECK_NE(padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], padded_image_folded_by_depth_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
  
//   float intermediate = 2*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size] 
//     + 3*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size-1] 
//     + 1*padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size+1];

//   BOOST_CHECK_EQUAL(intermediate, padded_image_folded_by_depth_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);
// }

// BOOST_AUTO_TEST_CASE( convolve_by_all1 )
// {
//   int center_one_dim_size = image_dims_[0]/2;
//   BOOST_CHECK_NE(padded_image_[center_one_dim_size][center_one_dim_size][center_one_dim_size], padded_image_folded_by_all1_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);

//   float value = 0.f;
//   for(int z_shift=-1;z_shift<=1;++z_shift){
//     for(int y_shift=-1;y_shift<=1;++y_shift){
//       for(int x_shift=-1;x_shift<=1;++x_shift){
// 	value += padded_image_[center_one_dim_size+x_shift][center_one_dim_size+y_shift][center_one_dim_size+z_shift];
//       }
//     }
//   }
//   BOOST_CHECK_EQUAL(value, padded_image_folded_by_all1_[center_one_dim_size][center_one_dim_size][center_one_dim_size]);

// }

BOOST_AUTO_TEST_SUITE_END()

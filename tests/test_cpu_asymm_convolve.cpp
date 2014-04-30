#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"

BOOST_FIXTURE_TEST_SUITE( convolution_works_with_asymm_kernels, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( cross_convolve )
{
  
  
  inplace_cpu_convolution(one_.data(), &image_dims_[0], 
			  asymm_cross_kernel_.data(),&asymm_kernel_dims_[0],
			  1);

  float sum_expected = std::accumulate(asymm_cross_kernel_.data(),asymm_cross_kernel_.data()+asymm_cross_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(one_.data(),one_.data()+one_.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);


}

BOOST_AUTO_TEST_CASE( one_convolve )
{
  
  
  inplace_cpu_convolution(one_.data(), &image_dims_[0], 
			  asymm_one_kernel_.data(),&asymm_kernel_dims_[0],
			  1);

  float sum_expected = std::accumulate(asymm_one_kernel_.data(),asymm_one_kernel_.data()+asymm_one_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(one_.data(),one_.data()+one_.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);


}

BOOST_AUTO_TEST_CASE( identity_convolve )
{
  
  
  inplace_cpu_convolution(one_.data(), &image_dims_[0], 
			  asymm_identity_kernel_.data(),&asymm_kernel_dims_[0],
			  1);

  float sum_expected = std::accumulate(asymm_identity_kernel_.data(),asymm_identity_kernel_.data()+asymm_identity_kernel_.num_elements(),0.f);
  float sum_received = std::accumulate(one_.data(),one_.data()+one_.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);


}

BOOST_AUTO_TEST_CASE( diagonal_convolve )
{
  
  using namespace multiviewnative;
  image_stack diagonal_kernel(asymm_kernel_dims_);
  for(int z_index = 0;z_index<int(diagonal_kernel.shape()[2]);++z_index){
    for(int y_index = 0;y_index<int(diagonal_kernel.shape()[1]);++y_index){
      for(int x_index = 0;x_index<int(diagonal_kernel.shape()[0]);++x_index){
	if(z_index == y_index && y_index == x_index)
	  diagonal_kernel[x_index][y_index][z_index] = 1.f;
	
	if(z_index == (int(diagonal_kernel.shape()[1]) -1 - y_index) && y_index == (int(diagonal_kernel.shape()[0]) -1 - x_index))
	  diagonal_kernel[x_index][y_index][z_index] = 1.f;
      }
    }
  }

  

  inplace_cpu_convolution(one_.data(), &image_dims_[0], 
			  diagonal_kernel.data(),&asymm_kernel_dims_[0],
			  1);

  float sum_expected = std::accumulate(diagonal_kernel.data(),diagonal_kernel.data()+diagonal_kernel.num_elements(),0.f);
  float sum_received = std::accumulate(one_.data(),one_.data()+one_.num_elements(),0.f);

  

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);


}

BOOST_AUTO_TEST_CASE( asymm_image_convolve )
{
  
  using namespace multiviewnative;
  image_stack diagonal_kernel(asymm_kernel_dims_);
  
  inplace_cpu_convolution(one_.data(), &image_dims_[0], 
			  diagonal_kernel.data(),&asymm_kernel_dims_[0],
			  1);

  float sum_expected = std::accumulate(diagonal_kernel.data(),diagonal_kernel.data()+diagonal_kernel.num_elements(),0.f);
  float sum_received = std::accumulate(one_.data(),one_.data()+one_.num_elements(),0.f);

  

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);


}
BOOST_AUTO_TEST_SUITE_END()

#include "tiff_fixtures.hpp"

using namespace multiviewnative;

static const tiff_stack kernel1_view_0("/dev/shm/libmultiview_data/kernel1_view_0.tif");

BOOST_AUTO_TEST_SUITE( convolution_works_with_kernel1_view_0 )

BOOST_AUTO_TEST_CASE( convolve_with_custom_one )
{

  tiff_stack local_kernel1(kernel1_view_0);
  std::vector<int> local_kernel1_dims(3);
  for(int i = 0;i<3;++i)
    local_kernel1_dims[i] = local_kernel1.stack_.shape()[i];

  unsigned max_dim = *std::max_element(local_kernel1_dims.begin(), local_kernel1_dims.end());
  std::vector<int> image_dims(3,3*max_dim);
  
  image_stack local_one(image_dims);
  local_one[image_dims[0]/2][image_dims[1]/2][image_dims[2]/2] = 1.f;
  
  inplace_cpu_convolution(local_one.data(), &image_dims[0], 
			  local_kernel1.stack_.data(),&local_kernel1_dims[0],
			  1);

  float sum_expected = std::accumulate(local_kernel1.stack_.data(),local_kernel1.stack_.data()+local_kernel1.stack_.num_elements(),0.f);
  float sum_received = std::accumulate(local_one.data(),local_one.data()+local_one.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);
}

BOOST_AUTO_TEST_CASE( convolve_with_multiple_custom_ones )
{

  tiff_stack local_kernel1(kernel1_view_0);
  std::vector<int> local_kernel1_dims(3);
  for(int i = 0;i<3;++i)
    local_kernel1_dims[i] = local_kernel1.stack_.shape()[i];

  unsigned max_dim = *std::max_element(local_kernel1_dims.begin(), local_kernel1_dims.end());
  std::vector<int> image_dims(3,5*max_dim);
  
  image_stack local_one(image_dims);
  local_one[image_dims[0]/2][image_dims[1]/2][image_dims[2]/2] = 1.f;

  local_one[1*image_dims[0]/4][image_dims[1]/2][image_dims[2]/2] = 1.f;
  local_one[3*image_dims[0]/4][image_dims[1]/2][image_dims[2]/2] = 1.f;

  local_one[image_dims[0]/2][1*image_dims[1]/4][image_dims[2]/2] = 1.f;
  local_one[image_dims[0]/2][3*image_dims[1]/4][image_dims[2]/2] = 1.f;

  local_one[image_dims[0]/2][image_dims[1]/2][1*image_dims[2]/4] = 1.f;
  local_one[image_dims[0]/2][image_dims[1]/2][3*image_dims[2]/4] = 1.f;
  
  inplace_cpu_convolution(local_one.data(), &image_dims[0], 
			  local_kernel1.stack_.data(),&local_kernel1_dims[0],
			  1);

  float sum_expected = std::accumulate(local_kernel1.stack_.data(),local_kernel1.stack_.data()+local_kernel1.stack_.num_elements(),0.f);
  float sum_received = std::accumulate(local_one.data(),local_one.data()+local_one.num_elements(),0.f);

  BOOST_CHECK_CLOSE(7*sum_expected, sum_received, .01f);

}
BOOST_AUTO_TEST_SUITE_END()

static const tiff_stack kernel2_view_0("/dev/shm/libmultiview_data/kernel2_view_0.tif");

BOOST_AUTO_TEST_SUITE( convolution_works_with_kernel2_view_0 )

BOOST_AUTO_TEST_CASE( convolve_with_custom_one )
{

  tiff_stack local_kernel2(kernel2_view_0);
  std::vector<int> local_kernel2_dims(3);
  for(int i = 0;i<3;++i)
    local_kernel2_dims[i] = local_kernel2.stack_.shape()[i];

  unsigned max_dim = *std::max_element(local_kernel2_dims.begin(), local_kernel2_dims.end());
  std::vector<int> image_dims(3,3*max_dim);
  
  image_stack local_one(image_dims);
  local_one[image_dims[0]/2][image_dims[1]/2][image_dims[2]/2] = 1.f;
  
  inplace_cpu_convolution(local_one.data(), &image_dims[0], 
			  local_kernel2.stack_.data(),&local_kernel2_dims[0],
			  1);

  float sum_expected = std::accumulate(local_kernel2.stack_.data(),local_kernel2.stack_.data()+local_kernel2.stack_.num_elements(),0.f);
  float sum_received = std::accumulate(local_one.data(),local_one.data()+local_one.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);
}

BOOST_AUTO_TEST_CASE( convolve_with_multiple_custom_ones )
{

  tiff_stack local_kernel2(kernel2_view_0);
  std::vector<int> local_kernel2_dims(3);
  for(int i = 0;i<3;++i)
    local_kernel2_dims[i] = local_kernel2.stack_.shape()[i];

  unsigned max_dim = *std::max_element(local_kernel2_dims.begin(), local_kernel2_dims.end());
  std::vector<int> image_dims(3,5*max_dim);
  
  image_stack local_one(image_dims);
  local_one[image_dims[0]/2][image_dims[1]/2][image_dims[2]/2] = 1.f;

  local_one[1*image_dims[0]/4][image_dims[1]/2][image_dims[2]/2] = 1.f;
  local_one[3*image_dims[0]/4][image_dims[1]/2][image_dims[2]/2] = 1.f;

  local_one[image_dims[0]/2][1*image_dims[1]/4][image_dims[2]/2] = 1.f;
  local_one[image_dims[0]/2][3*image_dims[1]/4][image_dims[2]/2] = 1.f;

  local_one[image_dims[0]/2][image_dims[1]/2][1*image_dims[2]/4] = 1.f;
  local_one[image_dims[0]/2][image_dims[1]/2][3*image_dims[2]/4] = 1.f;
  
  inplace_cpu_convolution(local_one.data(), &image_dims[0], 
			  local_kernel2.stack_.data(),&local_kernel2_dims[0],
			  1);

  float sum_expected = std::accumulate(local_kernel2.stack_.data(),local_kernel2.stack_.data()+local_kernel2.stack_.num_elements(),0.f);
  float sum_received = std::accumulate(local_one.data(),local_one.data()+local_one.num_elements(),0.f);

  BOOST_CHECK_CLOSE(7*sum_expected, sum_received, .01f);

}
BOOST_AUTO_TEST_SUITE_END()


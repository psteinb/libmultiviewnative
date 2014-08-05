#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
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
  
  multiviewnative::range expected_kernel_pos[3];
  for(int i = 0;i<3;++i)
    expected_kernel_pos[i] = multiviewnative::range(local_one.shape()[i]/2 - local_kernel1.stack_.shape()[i]/2,
						    local_one.shape()[i]/2 - local_kernel1.stack_.shape()[i]/2 + local_kernel1.stack_.shape()[i]);
  multiviewnative::image_stack_view kernel1_segment = local_one[ boost::indices[expected_kernel_pos[0]][expected_kernel_pos[1]][expected_kernel_pos[2]] ];
  multiviewnative::image_stack result = kernel1_segment;

  float l2norm = multiviewnative::l2norm(local_kernel1.stack_.data(), result.data(),result.num_elements());

  BOOST_CHECK_LT(l2norm, 1.e-5);
}

BOOST_AUTO_TEST_CASE( convolve_with_custom_one_discrete )
{

  tiff_stack local_kernel1(kernel1_view_0);
  std::vector<int> local_kernel1_dims(3);
  std::vector<int> offsets_kernel1(3);
  for(int i = 0;i<3;++i){
    local_kernel1_dims[i] = local_kernel1.stack_.shape()[i];
    offsets_kernel1[i] = local_kernel1.stack_.shape()[i]/2;
  }
  

  unsigned max_dim = *std::max_element(local_kernel1_dims.begin(), local_kernel1_dims.end());
  std::vector<int> image_dims(3,5*max_dim);
  
  image_stack local_one(image_dims);
  local_one[image_dims[0]/2][image_dims[1]/2][image_dims[2]/2] = 1.f;

  image_stack local_one_fft = local_one;

  inplace_cpu_convolution(local_one_fft.data(), &image_dims[0], 
			  local_kernel1.stack_.data(),&local_kernel1_dims[0],
			  1);

  // float sum_expected = std::accumulate(local_kernel1.stack_.data(),local_kernel1.stack_.data()+local_kernel1.stack_.num_elements(),0.f);
  // float sum_received = std::accumulate(local_one_fft.data(),local_one_fft.data()+local_one.num_elements(),0.f);

  image_stack local_one_discrete = local_one;
  multiviewnative::convolve(local_one,local_kernel1.stack_,local_one_discrete,offsets_kernel1);

  float l2norm = multiviewnative::l2norm(local_one_fft.data(), local_one_discrete.data(),local_one_fft.num_elements());

  BOOST_CHECK_LT(l2norm, 1.e-5);
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
			  2);

  float sum_expected = std::accumulate(local_kernel2.stack_.data(),local_kernel2.stack_.data()+local_kernel2.stack_.num_elements(),0.f);
  float sum_received = std::accumulate(local_one.data(),local_one.data()+local_one.num_elements(),0.f);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .001f);
multiviewnative::range expected_kernel_pos[3];
  for(int i = 0;i<3;++i)
    expected_kernel_pos[i] = multiviewnative::range(local_one.shape()[i]/2 - local_kernel2.stack_.shape()[i]/2,
						    local_one.shape()[i]/2 - local_kernel2.stack_.shape()[i]/2 + local_kernel2.stack_.shape()[i]);
  multiviewnative::image_stack_view kernel2_segment = local_one[ boost::indices[expected_kernel_pos[0]][expected_kernel_pos[1]][expected_kernel_pos[2]] ];
  multiviewnative::image_stack result = kernel2_segment;

  float l2norm = multiviewnative::l2norm(local_kernel2.stack_.data(), result.data(),result.num_elements());
  BOOST_CHECK_LT(l2norm, 1.e-5);
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
			  2);

  float sum_expected = std::accumulate(local_kernel2.stack_.data(),local_kernel2.stack_.data()+local_kernel2.stack_.num_elements(),0.f);
  float sum_received = std::accumulate(local_one.data(),local_one.data()+local_one.num_elements(),0.f);

  BOOST_CHECK_CLOSE(7*sum_expected, sum_received, .01f);

}
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( convolution_works_with_kernel2_view_0 )

BOOST_AUTO_TEST_CASE( convolve_with_custom_one_compare_to_gpu )
{

  tiff_stack local_kernel2(kernel2_view_0);
  std::vector<int> local_kernel2_dims(3);
  for(int i = 0;i<3;++i)
    local_kernel2_dims[i] = local_kernel2.stack_.shape()[i];

  unsigned max_dim = *std::max_element(local_kernel2_dims.begin(), local_kernel2_dims.end());
  std::vector<int> image_dims(3,3*max_dim);
  
  image_stack cpu_local_one(image_dims);
  cpu_local_one[image_dims[0]/2][image_dims[1]/2][image_dims[2]/2] = 1.f;
  
  inplace_cpu_convolution(cpu_local_one.data(), &image_dims[0], 
			  local_kernel2.stack_.data(),&local_kernel2_dims[0],
			  2);

  multiviewnative::range expected_kernel_pos[3];
  for(int i = 0;i<3;++i)
    expected_kernel_pos[i] = multiviewnative::range(cpu_local_one.shape()[i]/2 - local_kernel2.stack_.shape()[i]/2,
						    cpu_local_one.shape()[i]/2 - local_kernel2.stack_.shape()[i]/2 + local_kernel2.stack_.shape()[i]);
  multiviewnative::image_stack_view kernel2_segment = cpu_local_one[ boost::indices[expected_kernel_pos[0]][expected_kernel_pos[1]][expected_kernel_pos[2]] ];
  multiviewnative::image_stack cpu_result = kernel2_segment;

  float l2norm = multiviewnative::l2norm(local_kernel2.stack_.data(), cpu_result.data(),cpu_result.num_elements());
  BOOST_CHECK_LT(l2norm, 1.e-5);
  const int prec = std::cout.precision();
  std::cout.precision(4);
  std::cout << "cpu vs. expectation:\t" << l2norm << "\n";

  image_stack gpu_local_one(image_dims);
  gpu_local_one[image_dims[0]/2][image_dims[1]/2][image_dims[2]/2] = 1.f;
  
  inplace_gpu_convolution(gpu_local_one.data(), &image_dims[0], 
			  local_kernel2.stack_.data(),&local_kernel2_dims[0],
			  -1);

  kernel2_segment = gpu_local_one[ boost::indices[expected_kernel_pos[0]][expected_kernel_pos[1]][expected_kernel_pos[2]] ];
  multiviewnative::image_stack gpu_result = kernel2_segment;

  l2norm = multiviewnative::l2norm(local_kernel2.stack_.data(), gpu_result.data(),gpu_result.num_elements());
  BOOST_CHECK_LT(l2norm, 1.e-5);
  std::cout << "gpu vs. expectation:\t" << l2norm << "\n";
  
  l2norm = multiviewnative::l2norm(cpu_result.data(), gpu_result.data(),gpu_result.num_elements());
  BOOST_CHECK_LT(l2norm, 1.e-5);
  std::cout << "gpu vs. cpu        :\t" << l2norm << "\n";
}
BOOST_AUTO_TEST_SUITE_END()

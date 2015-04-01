#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_BATCHED_FFTs
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include "cufft_utils.cuh"
#include "cufft.h"
#include "cuda_helpers.cuh"

  using namespace multiviewnative;

BOOST_FIXTURE_TEST_SUITE(batched_fft,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(try_two) {

  

  image_stack reference = image_;
  cufftHandle ref_plan;
  HANDLE_CUFFT_ERROR(cufftPlan3d(&ref_plan, 
				 (int)reference.shape()[0],
				 (int)reference.shape()[1],
				 (int)reference.shape()[2], 
				 CUFFT_R2C));
  reference.resize(boost::extents[reference.shape()[0]][reference.shape()[1]][reference.shape()[2]/2 + 1]);
  std::vector<image_stack> input(4,reference);
  float* d_ref;
  const unsigned ref_size_byte = reference.num_elements()*sizeof(float);
  HANDLE_ERROR(cudaMalloc((void**)&(d_ref), ref_size_byte));
  HANDLE_ERROR(cudaMemcpy(d_ref, reference.data(), ref_size_byte,
			  cudaMemcpyHostToDevice));

  HANDLE_CUFFT_ERROR(
        cufftExecR2C(ref_plan, d_ref, (cufftComplex*)d_ref));

  HANDLE_ERROR(cudaMemcpy(reference.data(),d_ref, ref_size_byte,
			  cudaMemcpyDeviceToHost));

  //check my api
  //prepare space on device
  std::vector<float*> src_buffers(2);
  for (unsigned count = 0; count < src_buffers.size(); ++count){
    HANDLE_ERROR(cudaMalloc((void**)&(src_buffers[count]), ref_size_byte));
  }
  
  //batch transform
  std::vector<unsigned> shape(image_dims_.begin(), image_dims_.end());
  gpu::batched_fft_async2plans(input, 
			       shape, 
			       src_buffers, false);
  int matching = 0;
  for(unsigned count = 0; count < input.size(); ++count){
    matching += std::equal(reference.data(), reference.data() + reference.num_elements(), 
			   input[count].data());
  }

  BOOST_CHECK_EQUAL(matching, input.size());

  HANDLE_CUFFT_ERROR(cufftDestroy(ref_plan));

  HANDLE_ERROR(cudaFree(d_ref));
  for (unsigned count = 0; count < src_buffers.size(); ++count){
      HANDLE_ERROR(cudaFree(src_buffers[count]));
  }
}


BOOST_AUTO_TEST_SUITE_END()

#ifndef _CUDA_ND_FFT_H_
#define _CUDA_ND_FFT_H_
#include "cuda_helpers.cuh"
#include "cufft_utils.cuh"
#include "cufft.h"

// static void HandleCufftError(cufftResult_t err, const char* file, int line) {
//   if (err != CUFFT_SUCCESS) {
//     std::cerr << "cufftResult [" << err << "] in " << file << " at line "
//               << line << std::endl;
//     cudaDeviceReset();
//     exit(EXIT_FAILURE);
//   }
// }

// #define HANDLE_CUFFT_ERROR(err) (HandleCufftError(err, __FILE__, __LINE__))





/**
   \brief function that computes a r-2-c float32 FFT
   the function assumes that there has been space pre-allocated on device and
   that the data required has already been transferred

   \param[in] _d_src_buffer was already allocated to match the expected size of
   the FFT
   \param[in] _d_dest_buffer if non-zero, was already allocated to match the
   expected size of the FFT and if non-zero will be used as destionation buffer
   \param[in] _stack host-side nD image

   \return
   \retval

*/
void fft_excl_transfer_excl_alloc(const multiviewnative::image_stack& _stack,
                                  float* _d_src_buffer,
                                  float* _d_dest_buffer = 0,
                                  cufftHandle* _plan = 0) {

  if (!_plan) {
    _plan = new cufftHandle;

    HANDLE_CUFFT_ERROR(cufftPlan3d(_plan, (int)_stack.shape()[0],
                                   (int)_stack.shape()[1],
                                   (int)_stack.shape()[2], CUFFT_R2C));
    // HANDLE_CUFFT_ERROR(
    //     cufftSetCompatibilityMode(*_plan, CUFFT_COMPATIBILITY_NATIVE));
  }

  if (_d_dest_buffer)
    HANDLE_CUFFT_ERROR(
        cufftExecR2C(*_plan, _d_src_buffer, (cufftComplex*)_d_dest_buffer));
  else
    HANDLE_CUFFT_ERROR(
        cufftExecR2C(*_plan, _d_src_buffer, (cufftComplex*)_d_src_buffer));

  if (!_plan) {
    HANDLE_CUFFT_ERROR(cufftDestroy(*_plan));
    delete _plan;
  }

  HANDLE_ERROR(cudaDeviceSynchronize());
}

/**
   \brief function that computes a r-2-c float32 FFT
   the function assumes that there has been space pre-allocated on device

   \param[in] _d_src_buffer was already allocated to match the expected size of
   the FFT
   \param[in] _d_dest_buffer if non-zero, was already allocated to match the
   expected size of the FFT and if non-zero will be used as destionation buffer
   \param[in] _stack host-side nD image

   \return
   \retval

*/
void fft_incl_transfer_excl_alloc(const multiviewnative::image_stack& _stack,
                                  float* _d_src_buffer,
                                  float* _d_dest_buffer = 0,
                                  cufftHandle* _plan = 0) {

  unsigned stack_size_in_byte = _stack.num_elements() * sizeof(float);

  HANDLE_ERROR(cudaHostRegister((void*)_stack.data(), stack_size_in_byte,
                                cudaHostRegisterPortable));
  HANDLE_ERROR(cudaMemcpy(_d_src_buffer, _stack.data(), stack_size_in_byte,
                          cudaMemcpyHostToDevice));

  cufftComplex* dest_buffer = (cufftComplex*)_d_src_buffer;
  if (_d_dest_buffer) dest_buffer = (cufftComplex*)_d_dest_buffer;

  // perform FFT
  fft_excl_transfer_excl_alloc(_stack, _d_src_buffer, _d_dest_buffer, _plan);

  // to host
  // if (_d_dest_buffer)
    HANDLE_ERROR(cudaMemcpy((void*)_stack.data(), dest_buffer,
                            stack_size_in_byte, cudaMemcpyDeviceToHost));
  // else
  //   HANDLE_ERROR(cudaMemcpy((void*)_stack.data(), _d_src_buffer,
  //                           stack_size_in_byte, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaHostUnregister((void*)_stack.data()));
}

/**
   \brief function that computes a r-2-c float32 FFT

   \param[in] _stack host-side nD image
   \param[in] _d_dest_buffer if non-zero, was already allocated to match the
   expected size of the FFT and if non-zero will be used as destionation buffer

   \return
   \retval

*/
void fft_incl_transfer_incl_alloc(const multiviewnative::image_stack& _stack,
                                  float* _d_dest_buffer = 0,
                                  cufftHandle* _plan = 0) {

  float* src_buffer = 0;
  unsigned cufft_size_in_byte = multiviewnative::gpu::cufft_r2c_memory(
      &_stack.shape()[0],
      &_stack.shape()[0] + multiviewnative::image_stack::dimensionality);
  // alloc on device
  HANDLE_ERROR(cudaMalloc((void**)&(src_buffer), cufft_size_in_byte));

  fft_incl_transfer_excl_alloc(_stack, src_buffer, _d_dest_buffer, _plan);
  // to clean-up
  HANDLE_ERROR(cudaFree(src_buffer));
}

/**
  \brief calculatess the expected extra memory required by cufft

  \param[in] _shape vector that contains the dimensions

  \return
  \retval

 */
unsigned long cufft_3d_estimated_memory_consumption(
    const std::vector<unsigned>& _shape, cufftHandle* _plan = 0) {

  if (!_plan) {
    _plan = new cufftHandle;

    HANDLE_CUFFT_ERROR(cufftPlan3d(_plan, (int)_shape[0], (int)_shape[1],
                                   (int)_shape[2], CUFFT_R2C));

    // HANDLE_CUFFT_ERROR(
    //     cufftSetCompatibilityMode(*_plan, CUFFT_COMPATIBILITY_NATIVE));
  }

  size_t coarse_requirement = 0;
  HANDLE_CUFFT_ERROR(cufftEstimate3d((int)_shape[0], (int)_shape[1],
                                     (int)_shape[2], CUFFT_R2C,
                                     &coarse_requirement));

  size_t refined_requirement = 0;
  // TODO: this throws an error with cuda 6.5, research why?
  // http://stackoverflow.com/questions/26217721/cufft-invalid-value-in-cufftgetsize1d
  int cuda_api_version = 0;
  HANDLE_ERROR(cudaRuntimeGetVersion(&cuda_api_version));
  if (cuda_api_version <= 5050) {
    std::cout << "CALLING cufftGetSize3d\n";
    HANDLE_CUFFT_ERROR(cufftGetSize3d(*_plan, (int)_shape[0], (int)_shape[1],
                                      (int)_shape[2], CUFFT_R2C,
                                      &refined_requirement));
  }

  if (!_plan) {
    HANDLE_CUFFT_ERROR(cufftDestroy(*_plan));
    delete _plan;
  }

  return std::max(refined_requirement, coarse_requirement);
}

void batched_fft_synced(std::vector<multiviewnative::image_stack>& _stacks,
			float* _d_src_buffer,
			float* _d_dest_buffer = 0,
			cufftHandle* _plan = 0) {

  unsigned stack_size_in_byte = _stacks[0].num_elements() * sizeof(float);


  cufftComplex* dest_buffer = (cufftComplex*)_d_src_buffer;
  if (_d_dest_buffer) dest_buffer = (cufftComplex*)_d_dest_buffer;

  for( multiviewnative::image_stack& stack : _stacks ){
    HANDLE_ERROR(cudaHostRegister((void*)stack.data(), stack_size_in_byte,
				  cudaHostRegisterPortable));
    HANDLE_ERROR(cudaMemcpy(_d_src_buffer, stack.data(), stack_size_in_byte,
			    cudaMemcpyHostToDevice));
    
    HANDLE_CUFFT_ERROR(
		       cufftExecR2C(*_plan, _d_src_buffer, dest_buffer));


    HANDLE_ERROR(cudaMemcpy(stack.data(), 
			    (_d_dest_buffer ? _d_dest_buffer : _d_src_buffer),
			    stack_size_in_byte,
			    cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaHostUnregister((void*)stack.data()));
  }

  
}

void batched_fft_mapped(std::vector<multiviewnative::image_stack>& _stacks,
			 float* _d_src_buffer,
			 float* _d_dest_buffer = 0,
			 cufftHandle* _plan = 0) {

  unsigned stack_size_in_byte = _stacks[0].num_elements() * sizeof(float);
  cufftComplex* dest_buffer = (cufftComplex*)_d_src_buffer;
  if (_d_dest_buffer) dest_buffer = (cufftComplex*)_d_dest_buffer;
  cufftReal* current_src = 0;
  
  for( multiviewnative::image_stack& stack : _stacks ){
    HANDLE_ERROR(cudaHostRegister((void*)stack.data(), stack_size_in_byte,
				  cudaHostRegisterPortable));
    HANDLE_ERROR(cudaHostGetDevicePointer((void**)&current_src, (void*)stack.data(), 0));

    if (_d_dest_buffer){
      HANDLE_CUFFT_ERROR(
			 cufftExecR2C(*_plan, current_src, dest_buffer));
      HANDLE_ERROR(cudaMemcpy(stack.data(), 
			      _d_dest_buffer,
			      stack_size_in_byte,
			      cudaMemcpyDeviceToHost));
    }
    else
      HANDLE_CUFFT_ERROR(
			 cufftExecR2C(*_plan, current_src, (cufftComplex*)current_src));

    //    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaHostUnregister((void*)stack.data()));
  }
  


}

void batched_fft_async(std::vector<multiviewnative::image_stack>& _stacks,
		       float* _d_src_buffer,
		       float* _d_dest_buffer = 0,
		       cufftHandle* _plan = 0) {

  unsigned stack_size_in_byte = _stacks[0].num_elements() * sizeof(float);
  cufftComplex* dest_buffer = (cufftComplex*)_d_src_buffer;
  if (_d_dest_buffer) dest_buffer = (cufftComplex*)_d_dest_buffer;
  
  
  std::vector<cudaStream_t*> streams(_stacks.size());

  for( unsigned count = 0;count < _stacks.size();++count ){
    HANDLE_ERROR(cudaHostRegister((void*)_stacks[count].data(), 
				  stack_size_in_byte,
				  cudaHostRegisterPortable));
    streams[count] = new cudaStream_t;
    HANDLE_ERROR(cudaStreamCreate(streams[count]));
  
  }

  
  for( unsigned count = 0;count < _stacks.size();++count ){
    HANDLE_ERROR(cudaMemcpyAsync(_d_src_buffer,
				 _stacks[count].data(), 
				 stack_size_in_byte,
				 cudaMemcpyHostToDevice,
				 *streams[count]
				 ));
    
    HANDLE_CUFFT_ERROR(cufftSetStream(*_plan,				 
				      *streams[count] 
				      )
		       );

    HANDLE_CUFFT_ERROR(
		       cufftExecR2C(*_plan, _d_src_buffer, dest_buffer));
      
    HANDLE_ERROR(cudaMemcpyAsync(_stacks[count].data(), 
				 dest_buffer,
				 stack_size_in_byte,
				 cudaMemcpyDeviceToHost,
				 *streams[count])
		 );
    HANDLE_ERROR(cudaStreamSynchronize(*streams[count]));
  }
   
  for (unsigned count = 0;count < _stacks.size();++count){
    HANDLE_ERROR(cudaHostUnregister((void*)_stacks[count].data()));
    HANDLE_ERROR(cudaStreamDestroy(*streams[count]));
  }
  


}

//loosely based on nvidia-samples/6_Advanced/concurrentKernels/concurrentKernels.cu
void batched_fft_async2plans(std::vector<multiviewnative::image_stack>& _stacks,
			     std::vector<cufftHandle *>& _plans,
			     std::vector<float*>& _src_buffers,
			     bool register_input_stacks = true) {



  
  std::vector<cudaStream_t*> streams(_plans.size());
  for( unsigned count = 0;count < streams.size();++count ){
    streams[count] = new cudaStream_t;
    HANDLE_ERROR(cudaStreamCreate(streams[count]));
  }

  unsigned stack_size_in_byte = _stacks[0].num_elements() * sizeof(float);
  if(register_input_stacks){
    for( unsigned count = 0;count < _stacks.size();++count ){
      HANDLE_ERROR(cudaHostRegister((void*)_stacks[count].data(), 
				    stack_size_in_byte,
				    cudaHostRegisterPortable));
  
    }
  }

  std::vector<cudaEvent_t> before_plan_execution(_stacks.size());
  for( cudaEvent_t& e : before_plan_execution){
    HANDLE_ERROR(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
    
  float* d_buffer = 0;
  unsigned modulus_index = 0;
  for( unsigned count = 0;count < _stacks.size();++count ){

    modulus_index = count % streams.size();
    d_buffer = _src_buffers[count % streams.size()];
    
    HANDLE_ERROR(cudaMemcpyAsync(d_buffer,
				 _stacks[count].data(), 
				 stack_size_in_byte,
				 cudaMemcpyHostToDevice,
				 *streams[modulus_index]
				 ));

    
    HANDLE_CUFFT_ERROR(cufftSetStream(*_plans[modulus_index],				 
				      *streams[modulus_index] )
		       );

    HANDLE_ERROR(cudaEventRecord(before_plan_execution[count],*streams[modulus_index]));
    if(count>0)
      HANDLE_ERROR(cudaStreamWaitEvent(*streams[modulus_index], before_plan_execution[count-1],0));
    
    HANDLE_CUFFT_ERROR(
		       cufftExecR2C(*_plans[modulus_index], d_buffer, (cufftComplex*)d_buffer));
    
				 
    HANDLE_ERROR(cudaMemcpyAsync(_stacks[count].data(), 
				 d_buffer,
				 stack_size_in_byte,
				 cudaMemcpyDeviceToHost,
				 *streams[modulus_index])
		 );
    
  }
   
  //clean-up
  for (unsigned count = 0;count < streams.size();++count){
    HANDLE_ERROR(cudaStreamSynchronize(*streams[count]));
    HANDLE_ERROR(cudaStreamDestroy(*streams[count]));
  }
  
  for (unsigned count = 0;count < _stacks.size();++count){
    HANDLE_ERROR(cudaEventDestroy(before_plan_execution[count]));
    if(register_input_stacks)
      HANDLE_ERROR(cudaHostUnregister((void*)_stacks[count].data()));
  }
  

}

//loosely based on nvidia-samples/./0_Simple/UnifiedMemoryStreams/UnifiedMemoryStreams.cu
void batched_fft_managed(std::vector<float*>& _stacks,
			 cufftHandle* _plan) {

  HANDLE_ERROR(cudaDeviceSynchronize());
  for( unsigned count = 0;count < _stacks.size();++count ){
    HANDLE_CUFFT_ERROR(
		       cufftExecR2C(*_plan, _stacks[count], (cufftComplex*)_stacks[count]));
  }
  HANDLE_ERROR(cudaDeviceSynchronize());
}
#endif /* _CUDA_ND_FFT_H_ */

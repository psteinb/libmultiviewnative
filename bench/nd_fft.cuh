#ifndef _CUDA_ND_FFT_H_		
#define _CUDA_ND_FFT_H_
#include "cuda_helpers.cuh"
#include "cufft.h"

static void HandleCufftError( cufftResult_t err,
                         const char *file,
                         int line ) {
    if (err != CUFFT_SUCCESS) {
      std::cerr << "cufftResult [" << err << "] in " << file << " at line " << line << std::endl;
      cudaDeviceReset();
      exit( EXIT_FAILURE );
    }
}

#define HANDLE_CUFFT_ERROR( err ) (HandleCufftError( err, __FILE__, __LINE__ ))

/**
   \brief calculates the expected memory consumption for an inplace r-2-c transform according to 
   http://docs.nvidia.com/cuda/cufft/index.html#data-layout
   
   \param[in] _shape std::vector that contains the dimensions
   
   \return numbe of bytes consumed
   \retval 
   
*/
unsigned long cufft_inplace_r2c_memory(const std::vector<unsigned>& _shape){
  
  unsigned long start = 1;
  unsigned long value = std::accumulate(_shape.begin(), 
					_shape.end()-1, 
					start, 
					std::multiplies<unsigned long>());
  value *= (std::floor(_shape.back()/2.) + 1)*sizeof(cufftComplex);
  return value;
}

/**
   \brief calculates the expected memory consumption for an inplace r-2-c transform according to 
   http://docs.nvidia.com/cuda/cufft/index.html#data-layout
   
   \param[in] _shape_begin begin iterator of shape array
   \param[in] _shape_end end iterator of shape array
   
   \return numbe of bytes consumed
   \retval 
   
*/
template <typename Iter>
unsigned long cufft_inplace_r2c_memory(Iter _shape_begin, Iter _shape_end){
  
  unsigned long start = 1;
  unsigned long value = std::accumulate(_shape_begin, 
					_shape_end-1, 
					start, 
					std::multiplies<unsigned long>());
  value *= (std::floor(*(_shape_end-1)/2.) + 1)*sizeof(cufftComplex);
  return value;
}

/**
   \brief function that computes a r-2-c float32 FFT
   the function assumes that there has been space pre-allocated on device
   
   \param[in] _d_prealloc_buffer was already allocated to match the expected size of the FFT
   \param[in] _stack host-side nD image

   \return 
   \retval 
   
*/
void inplace_fft_incl_transfer_excl_alloc(const multiviewnative::image_stack& _stack, 
			       float* _d_prealloc_buffer){
  
  unsigned stack_size_in_byte = _stack.num_elements()*sizeof(float);
  HANDLE_ERROR( cudaHostRegister((void*)_stack.data()   , stack_size_in_byte , cudaHostRegisterPortable) );
  HANDLE_ERROR( cudaMemcpy(_d_prealloc_buffer, _stack.data()   , stack_size_in_byte , cudaMemcpyHostToDevice) );

  //to device
  cufftHandle transform_plan_;

  HANDLE_CUFFT_ERROR(cufftPlan3d(&transform_plan_, 
				 (int)_stack.shape()[0], 
				 (int)_stack.shape()[1], 
				 (int)_stack.shape()[2], 
				 CUFFT_R2C));
  HANDLE_CUFFT_ERROR(cufftSetCompatibilityMode(transform_plan_,CUFFT_COMPATIBILITY_NATIVE));
  
  HANDLE_CUFFT_ERROR(cufftExecR2C(transform_plan_, 
				  _d_prealloc_buffer, 
				  (cufftComplex *)_d_prealloc_buffer));

  HANDLE_CUFFT_ERROR( cufftDestroy(transform_plan_) );

  //to host
  HANDLE_ERROR( cudaMemcpy((void*)_stack.data()   , _d_prealloc_buffer, stack_size_in_byte , cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaHostUnregister((void*)_stack.data()) );
  HANDLE_ERROR( cudaDeviceSynchronize() );
	
}

/**
   \brief function that computes a r-2-c float32 FFT
   the function assumes that there has been space pre-allocated on device and that the data required has already been transferred
   
   \param[in] _d_prealloc_buffer was already allocated to match the expected size of the FFT
   \param[in] _stack host-side nD image

   \return 
   \retval 
   
*/
void fft_excl_transfer_excl_alloc(const multiviewnative::image_stack& _stack, 
				  float* _d_prealloc_buffer,
				  float* _d_prealloc_target = 0){
  
  

  //to device
  cufftHandle transform_plan_;

  HANDLE_CUFFT_ERROR(cufftPlan3d(&transform_plan_, 
				 (int)_stack.shape()[0], 
				 (int)_stack.shape()[1], 
				 (int)_stack.shape()[2], 
				 CUFFT_R2C));
  HANDLE_CUFFT_ERROR(cufftSetCompatibilityMode(transform_plan_,CUFFT_COMPATIBILITY_NATIVE));
  
  if(_d_prealloc_target)
    HANDLE_CUFFT_ERROR(cufftExecR2C(transform_plan_, 
				    _d_prealloc_buffer, 
				    (cufftComplex *)_d_prealloc_target));
  else
    HANDLE_CUFFT_ERROR(cufftExecR2C(transform_plan_, 
				    _d_prealloc_buffer, 
				    (cufftComplex *)_d_prealloc_buffer));

  HANDLE_CUFFT_ERROR( cufftDestroy(transform_plan_) );

  
  HANDLE_ERROR( cudaDeviceSynchronize() );	
}

/**
   \brief function that computes a r-2-c float32 FFT
      
   \param[in] _stack host-side nD image
   
   \return 
   \retval 
   
*/
void inplace_fft_incl_transfer_incl_alloc(const multiviewnative::image_stack& _stack){
  
  float* d_buffer = 0;
  //on host
  unsigned stack_size_in_byte = _stack.num_elements()*sizeof(float);
  HANDLE_ERROR( cudaHostRegister((void*)_stack.data()   , stack_size_in_byte , cudaHostRegisterPortable) );
  
  unsigned cufft_size_in_byte = cufft_inplace_r2c_memory(&_stack.shape()[0], 
							 &_stack.shape()[0] + multiviewnative::image_stack::dimensionality);
    //alloc on device
  HANDLE_ERROR( cudaMalloc( (void**)&(d_buffer), cufft_size_in_byte ) );
  
  //to device
  HANDLE_ERROR( cudaMemcpy(d_buffer, (void*)_stack.data()   , stack_size_in_byte , cudaMemcpyHostToDevice) );

    //compute
  cufftHandle transform_plan_;

  HANDLE_CUFFT_ERROR(cufftPlan3d(&transform_plan_, 
				 (int)_stack.shape()[0], 
				 (int)_stack.shape()[1], 
				 (int)_stack.shape()[2], 
				 CUFFT_R2C));
  HANDLE_CUFFT_ERROR(cufftSetCompatibilityMode(transform_plan_,CUFFT_COMPATIBILITY_NATIVE));
  
  HANDLE_CUFFT_ERROR(cufftExecR2C(transform_plan_, 
				  d_buffer, 
				  (cufftComplex *)d_buffer));

  HANDLE_CUFFT_ERROR( cufftDestroy(transform_plan_) );

  HANDLE_ERROR( cudaMemcpy((void*)_stack.data()   , d_buffer, stack_size_in_byte , cudaMemcpyDeviceToHost) );
  //to clean-up
  HANDLE_ERROR( cudaFree( d_buffer ) );
  HANDLE_ERROR( cudaHostUnregister((void*)_stack.data()) );
  HANDLE_ERROR( cudaDeviceSynchronize() );	
}


/**
  \brief calculatess the expected extra memory required by cufft

  \param[in] _shape vector that contains the dimensions

  \return 
  \retval 

 */
unsigned long cufft_3d_estimated_memory_consumption(const std::vector<unsigned>& _shape){
  
  cufftHandle transform_plan_;

  HANDLE_CUFFT_ERROR(cufftPlan3d(&transform_plan_, 
  				 (int)_shape[0], 
  				 (int)_shape[1], 
  				 (int)_shape[2], 
  				 CUFFT_R2C));

  HANDLE_CUFFT_ERROR(cufftSetCompatibilityMode(transform_plan_,CUFFT_COMPATIBILITY_NATIVE));


  size_t coarse_requirement = 0;
  HANDLE_CUFFT_ERROR( cufftEstimate3d((int)_shape[0], (int)_shape[1], (int)_shape[2], CUFFT_R2C, &coarse_requirement) );

  size_t refined_requirement = 0;
  // TODO: this throws an error with cuda 6.5, research why?
  // http://stackoverflow.com/questions/26217721/cufft-invalid-value-in-cufftgetsize1d
  int cuda_api_version = 0;
  HANDLE_ERROR(cudaRuntimeGetVersion(&cuda_api_version));
  if(cuda_api_version <= 5050){
    std::cout << "CALLING cufftGetSize3d\n";
    HANDLE_CUFFT_ERROR(  cufftGetSize3d(transform_plan_,
					(int)_shape[0], 
					(int)_shape[1], 
					(int)_shape[2], 
					CUFFT_R2C, 
					&refined_requirement) );
  }
  
  HANDLE_CUFFT_ERROR( cufftDestroy(transform_plan_) );

  return std::max(refined_requirement, coarse_requirement);
}




#endif /* _CUDA_ND_FFT_H_ */

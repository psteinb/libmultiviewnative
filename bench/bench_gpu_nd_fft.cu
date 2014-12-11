#define __BENCH_GPU_DECONVOLVE_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include "boost/program_options.hpp" 
#include "synthetic_data.hpp"
#include "cuda_helpers.cuh"
#include "cufft.h"

#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;

namespace po = boost::program_options;

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
   \brief function that computes a r-2-c float32 FFT
   the function assumes that there has been space allocated on device
   
   \param[in] _d_prealloc_buffer was already allocated to match the expected size of the FFT
   
   \return 
   \retval 
   
*/
void inplace_fft_with_transfer(const multiviewnative::image_stack& _stack, 
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
   the function assumes that there has been space allocated on device and that the data required has already been transferred
   
   \param[in] _d_prealloc_buffer was already allocated to match the expected size of the FFT
   
   \return 
   \retval 
   
*/
void inplace_fft_without_transfer(const multiviewnative::image_stack& _stack, 
			       float* _d_prealloc_buffer){
  
  

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
  // TODO: this throws an error, research why?
  // HANDLE_CUFFT_ERROR(  cufftGetSize3d(transform_plan_,
  // 				      (int)_shape[0], 
  // 				      (int)_shape[1], 
  // 				      (int)_shape[2], 
  // 				      CUFFT_R2C, 
  // 				      &refined_requirement) );

  
  HANDLE_CUFFT_ERROR( cufftDestroy(transform_plan_) );

  return std::max(refined_requirement, coarse_requirement);
}

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


int main(int argc, char *argv[])
{

  bool verbose = false;
  bool with_transfers = false;
  bool with_allocation = false;
  
  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("verbose,v", "print lots of information in between")
    ("stack_dimensions,s", po::value<std::string>(&stack_dims)->default_value("512x512x64"), "HxWxD of synthetic stacks to generate")
    ("with_transfers,w", "include host-device transfers in timings" )
    ("with_allocation,a", "include host-device memory allocation" )
    ("repeats,r", po::value<int>(&num_repeats)->default_value(10), "number of repetitions per measurement")
    // ("input-files", po::value<std::vector<std::string> >()->composing(), "")
    ;

  po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);

  po::notify(vm); 

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  verbose = vm.count("verbose");
  with_transfers = vm.count("with_transfers");
  with_allocation = vm.count("with_allocation");
  
  

  std::vector<unsigned> numeric_stack_dims(3);
  numeric_stack_dims[0] = 512;
  numeric_stack_dims[1] = 512;
  numeric_stack_dims[2] = 64;
  
  split<'x'>(stack_dims,numeric_stack_dims);
  
  if(verbose){
    std::cout << "received dimensions: ";
    for (unsigned i = 0; i < numeric_stack_dims.size(); ++i)
      {
	std::cout << numeric_stack_dims[i] << " ";
      }
    std::cout << "\n";
  }
  
  int device_id = selectDeviceWithHighestComputeCapability();
  HANDLE_ERROR( cudaSetDevice(device_id));
  unsigned long cufft_extra_space = cufft_3d_estimated_memory_consumption(numeric_stack_dims);
  unsigned long cufft_data_size = cufft_inplace_r2c_memory(numeric_stack_dims);
  unsigned long memory_available_on_device = getAvailableGMemOnCurrentDevice();

  float exp_mem_mb = (cufft_extra_space+cufft_data_size)/float(1 << 20);
  float av_mem_mb = memory_available_on_device/float(1 << 20);

  if(exp_mem_mb>av_mem_mb){
    std::cerr << "not enough memory available on device, needed " << exp_mem_mb 
	      << " MB (data only: "<< cufft_data_size/float(1<< 20) 
	      <<" MB), available: " << av_mem_mb << " MB\n";
    return 1;
  } else {
    if(verbose)
      std::cout << "cufft memory estimate: needed " << exp_mem_mb 
		<< " MB (data only: "<< cufft_data_size/float(1<< 20) 
		<<" MB), available: " << av_mem_mb << " MB\n";
  }

  multiviewnative::image_kernel_data data(numeric_stack_dims);
  if(verbose)
    data.info();
  
  
  std::vector<cpu_times> durations(num_repeats);

  double time_ms = 0.f;
  
  if(!with_allocation){
    
    float* d_preallocated_buffer = 0;
    unsigned size_in_byte_ = std::accumulate(numeric_stack_dims.begin(), 
					     numeric_stack_dims.end()-1, 1, 
					     std::multiplies<unsigned>())*(std::floor(numeric_stack_dims.back()/2.) + 1)*sizeof(cufftComplex);

    HANDLE_ERROR( cudaMalloc( (void**)&(d_preallocated_buffer), size_in_byte_ ) );

    if(with_transfers){
      //warm-up
      inplace_fft_with_transfer(data.stack_,
				  d_preallocated_buffer );
      
      for(int r = 0;r<num_repeats;++r){
	cpu_timer timer;
	inplace_fft_with_transfer(data.stack_,
				  d_preallocated_buffer );
	durations[r] = timer.elapsed();

	time_ms += double(durations[r].system + durations[r].user)/1e6;
	if(verbose){
	  std::cout << " took " << double(durations[r].system + durations[r].user)/1e6 << " ms\n";
	}
      }
    } else {
      
      unsigned stack_size_in_byte = data.stack_.num_elements()*sizeof(float);
      HANDLE_ERROR( cudaHostRegister((void*)data.stack_.data()   , stack_size_in_byte , cudaHostRegisterPortable) );
      HANDLE_ERROR( cudaMemcpy(d_preallocated_buffer, data.stack_.data()   , stack_size_in_byte , cudaMemcpyHostToDevice) );

      for(int r = 0;r<num_repeats;++r){
	cpu_timer timer;
	inplace_fft_without_transfer(data.stack_,
				  d_preallocated_buffer );
	durations[r] = timer.elapsed();

	time_ms += double(durations[r].system + durations[r].user)/1e6;
	if(verbose){
	  std::cout << " took " << double(durations[r].system + durations[r].user)/1e6 << " ms\n";
	}
      }

      //to host
      HANDLE_ERROR( cudaMemcpy((void*)data.stack_.data()   , d_preallocated_buffer, stack_size_in_byte , cudaMemcpyDeviceToHost) );
      HANDLE_ERROR( cudaHostUnregister((void*)data.stack_.data()) );
    }
    
    HANDLE_ERROR( cudaFree( d_preallocated_buffer ) );
  } else {
    //timing should include allocation

  }

  std::cout << "[bench_gpu_nd_fft] " << time_ms << " ms\n";

  

  return 0;
}

#define __BENCH_GPU_DECONVOLVE_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include "boost/program_options.hpp" 
#include "synthetic_data.hpp"

#include "cuda_profiler_api.h"
#include "nd_fft.cuh"
// #include "cufft.h"

#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;

namespace po = boost::program_options;



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
    ("with_transfers,t", "include host-device transfers in timings" )
    ("with_allocation,a", "include host-device memory allocation in timings" )
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

  
  std::vector<unsigned> numeric_stack_dims;
  split<'x'>(stack_dims,numeric_stack_dims);
  
  if(verbose){
    std::cout << "received "<< numeric_stack_dims.size() <<" dimensions: ";
    for (unsigned i = 0; i < numeric_stack_dims.size(); ++i)
      {
	std::cout << numeric_stack_dims[i] << " ";
      }
    std::cout << "\n";
  }

  if(numeric_stack_dims.size()!=3){
    std::cerr << ">> " << numeric_stack_dims.size() << "-D data, not supported yet!\n";
    return 1;
  }
  
  int device_id = selectDeviceWithHighestComputeCapability();
  HANDLE_ERROR( cudaSetDevice(device_id));
  unsigned long cufft_extra_space = cufft_3d_estimated_memory_consumption(numeric_stack_dims);
  unsigned long cufft_data_size = cufft_inplace_r2c_memory(numeric_stack_dims);
  unsigned long data_size_byte = std::accumulate(numeric_stack_dims.begin(), numeric_stack_dims.end(), 1u, std::multiplies<unsigned long>())*sizeof(float);
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
    unsigned size_in_byte_ = cufft_inplace_r2c_memory(numeric_stack_dims);

    HANDLE_ERROR( cudaMalloc( (void**)&(d_preallocated_buffer), size_in_byte_ ) );

    if(with_transfers){
      //warm-up
      inplace_fft_incl_transfer_excl_alloc(data.stack_,
				d_preallocated_buffer );

      cudaProfilerStart();
      for(int r = 0;r<num_repeats;++r){
	cpu_timer timer;
	inplace_fft_incl_transfer_excl_alloc(data.stack_,
					     d_preallocated_buffer );
	durations[r] = timer.elapsed();

	time_ms += double(durations[r].system + durations[r].user)/1e6;
	if(verbose){
	  std::cout << r << "\t" << double(durations[r].system + durations[r].user)/1e6 << " ms\n";
	}
      }
      cudaProfilerStop();

    } else {
      
      unsigned stack_size_in_byte = data.stack_.num_elements()*sizeof(float);
      HANDLE_ERROR( cudaHostRegister((void*)data.stack_.data()   , stack_size_in_byte , cudaHostRegisterPortable) );
      HANDLE_ERROR( cudaMemcpy(d_preallocated_buffer, data.stack_.data()   , stack_size_in_byte , cudaMemcpyHostToDevice) );
      //warm-up
      inplace_fft_excl_transfer_excl_alloc(data.stack_,
					   d_preallocated_buffer );
      
      cudaProfilerStart();
      for(int r = 0;r<num_repeats;++r){
	cpu_timer timer;
	inplace_fft_excl_transfer_excl_alloc(data.stack_,
					     d_preallocated_buffer );
	durations[r] = timer.elapsed();

	time_ms += double(durations[r].system + durations[r].user)/1e6;
	if(verbose){
	  std::cout << r << "\t" << double(durations[r].system + durations[r].user)/1e6 << " ms\n";

	}
      }
      cudaProfilerStop();

      //to host
      HANDLE_ERROR( cudaMemcpy((void*)data.stack_.data()   , d_preallocated_buffer, stack_size_in_byte , cudaMemcpyDeviceToHost) );
      HANDLE_ERROR( cudaHostUnregister((void*)data.stack_.data()) );
    }
    
    HANDLE_ERROR( cudaFree( d_preallocated_buffer ) );
  } else {
    //wamr-up
    inplace_fft_incl_transfer_incl_alloc(data.stack_);
    //timing should include allocation, which requires including transfers
    cudaProfilerStart();
      for(int r = 0;r<num_repeats;++r){
	cpu_timer timer;
	inplace_fft_incl_transfer_incl_alloc(data.stack_);
	durations[r] = timer.elapsed();

	time_ms += double(durations[r].system + durations[r].user)/1e6;
	if(verbose){
	  std::cout << r << "\t" << double(durations[r].system + durations[r].user)/1e6 << " ms\n";
	}
      }
      cudaProfilerStop();

  }

  std::string device_name = get_cuda_device_name(device_id);
  std::replace(device_name.begin(), device_name.end(), ' ', '_');

  
  std::cout << device_name << "\t"
	    << ( (with_allocation) ? "incl_alloc" : "excl_alloc") << "\t" 
	    << ( (with_transfers) ? "incl_tx" : "excl_tx") << "\t" 
	    << num_repeats <<"\t" 
	    << time_ms << "\t" 
	    << stack_dims << "\t" 
	    << data_size_byte/float(1 << 20) << "\t" 
	    << exp_mem_mb
	    << "\n";

  

  return 0;
}

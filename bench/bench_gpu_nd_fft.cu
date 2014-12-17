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
  bool out_of_place = false;
  bool use_global_plan = false;
  cufftHandle* global_plan = 0;
  
  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("verbose,v", "print lots of information in between")
    ("stack_dimensions,s", po::value<std::string>(&stack_dims)->default_value("512x512x64"), "HxWxD of synthetic stacks to generate")
    ("with_transfers,t", "include host-device transfers in timings" )
    ("global_plan,g", "use a global plan, rather than creating a plan everytime a transformation is performed" )
    ("out-of-place,o", "perform out-of-place transforms" )
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
  out_of_place = vm.count("out-of-place");
  use_global_plan = vm.count("global_plan");

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
  unsigned long cufft_data_size = cufft_r2c_memory(numeric_stack_dims);
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

  
  if(use_global_plan){
    global_plan = new cufftHandle;

    HANDLE_CUFFT_ERROR(cufftPlan3d(global_plan, 
				   (int)numeric_stack_dims[0], 
				   (int)numeric_stack_dims[1], 
				   (int)numeric_stack_dims[2], 
				   CUFFT_R2C));

    HANDLE_CUFFT_ERROR(cufftSetCompatibilityMode(*global_plan,CUFFT_COMPATIBILITY_NATIVE));
  }

  multiviewnative::image_kernel_data data(numeric_stack_dims);
  if(verbose){
    std::cout << "[config]\t" 
	      << ( (with_allocation) ? "incl_alloc" : "excl_alloc") << " " 
	      << ( (with_transfers) ? "incl_tx" : "excl_tx") << " " 
	      << ( (out_of_place) ? "out-of-place" : "inplace") << " " 
      	      << ( (use_global_plan) ? "global plans" : "local plans") << " " 
	      << "\n";
    data.info();
  }
  
  std::vector<cpu_times> durations(num_repeats);

  double time_ms = 0.f;
  float* d_dest_buffer = 0; 
  const unsigned fft_size_in_byte_ = cufft_r2c_memory(numeric_stack_dims); 
  if(out_of_place)
    HANDLE_ERROR( cudaMalloc( (void**)&(d_dest_buffer), fft_size_in_byte_ ) );

  if(!with_allocation){
    
    float* d_src_buffer = 0; 
      
    if(out_of_place)
      HANDLE_ERROR( cudaMalloc( (void**)&(d_src_buffer), data_size_byte ) );
    else
      HANDLE_ERROR( cudaMalloc( (void**)&(d_src_buffer), fft_size_in_byte_ ) );

    if(with_transfers){
      //warm-up
      fft_incl_transfer_excl_alloc(data.stack_,
				   d_src_buffer ,
				   out_of_place ? d_dest_buffer : 0,
				   use_global_plan ? global_plan : 0);

      cudaProfilerStart();
      for(int r = 0;r<num_repeats;++r){
	cpu_timer timer;
	fft_incl_transfer_excl_alloc(data.stack_,
				     d_src_buffer ,
				     out_of_place ? d_dest_buffer : 0,
				     use_global_plan ? global_plan : 0);
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
      HANDLE_ERROR( cudaMemcpy(d_src_buffer, data.stack_.data()   , stack_size_in_byte , cudaMemcpyHostToDevice) );
      //warm-up
      fft_excl_transfer_excl_alloc(data.stack_,
				   d_src_buffer ,
				   out_of_place ? d_dest_buffer : 0,
				   use_global_plan ? global_plan : 0);
      
      cudaProfilerStart();
      for(int r = 0;r<num_repeats;++r){
	cpu_timer timer;
	fft_excl_transfer_excl_alloc(data.stack_,
				     d_src_buffer,
				     out_of_place ? d_dest_buffer : 0 ,
				     use_global_plan ? global_plan : 0);
	durations[r] = timer.elapsed();

	time_ms += double(durations[r].system + durations[r].user)/1e6;
	if(verbose){
	  std::cout << r << "\t" << double(durations[r].system + durations[r].user)/1e6 << " ms\n";

	}
      }
      cudaProfilerStop();

      //to host
      HANDLE_ERROR( cudaMemcpy((void*)data.stack_.data()   , d_src_buffer, stack_size_in_byte , cudaMemcpyDeviceToHost) );
      HANDLE_ERROR( cudaHostUnregister((void*)data.stack_.data()) );
      
    }
    
    HANDLE_ERROR( cudaFree( d_src_buffer ) );
    
  } else {
    with_transfers = true;
    //warm-up
    fft_incl_transfer_incl_alloc(data.stack_,
				 out_of_place ? d_dest_buffer : 0,
				 use_global_plan ? global_plan : 0);
    //timing should include allocation, which requires including transfers
    cudaProfilerStart();
    for(int r = 0;r<num_repeats;++r){
      cpu_timer timer;
      fft_incl_transfer_incl_alloc(data.stack_,
				   out_of_place ? d_dest_buffer : 0,
				   use_global_plan ? global_plan : 0);
      durations[r] = timer.elapsed();

      time_ms += double(durations[r].system + durations[r].user)/1e6;
      if(verbose){
	std::cout << r << "\t" << double(durations[r].system + durations[r].user)/1e6 << " ms\n";
      }
    }
    cudaProfilerStop();

  }

  if(out_of_place)
    HANDLE_ERROR( cudaFree( d_dest_buffer ) );
  
  if(use_global_plan){
    HANDLE_CUFFT_ERROR( cufftDestroy(*global_plan) );
    delete global_plan;
  }

  std::string device_name = get_cuda_device_name(device_id);
  std::replace(device_name.begin(), device_name.end(), ' ', '_');

  
  std::cout << device_name << " "
	    << ( (with_allocation) ? "incl_alloc" : "excl_alloc") << " " 
	    << ( (with_transfers) ? "incl_tx" : "excl_tx") << " " 
	    << ( (out_of_place) ? "out-of-place" : "inplace") << " " 
	    << num_repeats <<" " 
	    << time_ms << " " 
	    << stack_dims << " " 
	    << data_size_byte/float(1 << 20) << " " 
	    << exp_mem_mb
	    << "\n";

  

  return 0;
}

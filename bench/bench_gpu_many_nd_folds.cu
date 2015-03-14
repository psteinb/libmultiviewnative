#define __BENCH_GPU_DECONVOLVE_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include "boost/program_options.hpp"
#include "synthetic_data.hpp"
//#include "cpu_nd_fft.hpp"
#include "multiviewnative.h"
//#include "fftw_interface.h"

#include "logging.hpp"

#include <boost/chrono.hpp>
#include <boost/thread.hpp>


#include "gpu_convolve.cuh"
#include "padd_utils.h"
#include "gpu_nd_fft.cuh"
#include "cufft_utils.cuh"


namespace mvn = multiviewnative;
typedef mvn::no_padd<mvn::image_stack> stack_padding;
typedef mvn::inplace_3d_transform_on_device<imageType>
    device_transform;
typedef mvn::gpu_convolve<stack_padding, imageType, unsigned>
    device_convolve;


template <typename Container>
void inplace_gpu_batched_fold(std::vector<Container>& _data){
  
  std::vector<mvn::image_stack> forwarded_kernels(_data.size());

  std::vector<int> reshaped;


  for (int v = 0; v < _data.size(); ++v) {

    stack_padding local_padding(&_data[v].stack_shape_[0],
				      &_data[v].kernel_shape_[0]);

    forwarded_kernels[v].resize(local_padding.extents_);
    local_padding.wrapped_insert_at_offsets(_data[v].kernel_, forwarded_kernels[v]);

    //prepare for fft
    reshaped = multiviewnative::gpu::cufft_r2c_shape(forwarded_kernels[v].shape(),forwarded_kernels[v].shape() + 3);
    forwarded_kernels[v].resize(reshaped);
    HANDLE_ERROR(cudaHostRegister((void*)forwarded_kernels[v].data(), 
				    forwarded_kernels[v].num_elements()*sizeof(float),
				    cudaHostRegisterPortable));
  }

  unsigned long reshaped_buffer_byte = forwarded_kernels[0].num_elements()*sizeof(float);
  
  std::vector<device_convolve*> image_folds(_data.size(),0);
  for (int v = 0; v < _data.size(); ++v) {
    image_folds[v] = new device_convolve(_data[v].stack_.data(),
				      &(_data[v].stack_shape_[0]),
				      &(_data[v].kernel_shape_[0])
				      );
		     
  }

  //creating the plans
  std::vector<cufftHandle *> plans(2 //number of copy engines
				   );
  for (unsigned count = 0; count < plans.size(); ++count) {

      plans[count] = new cufftHandle;
      HANDLE_CUFFT_ERROR(cufftPlan3d(plans[count],                 //
				     (int)_data[0].stack_shape_[0], //
				     (int)_data[0].stack_shape_[1], //
				     (int)_data[0].stack_shape_[2], //
				     CUFFT_R2C)                    //
			 );
      
  }

  //requesting space on device
  std::vector<float*> src_buffers(plans.size(),0);
  for (unsigned count = 0; count < src_buffers.size(); ++count){
    HANDLE_ERROR(cudaMalloc((void**)&(src_buffers[count]), reshaped_buffer_byte));
    
  }

  
  //forward all kernels
  batched_fft_async2plans(forwarded_kernels,plans,src_buffers,false);
    
  //perform convolution
  std::vector<cudaStream_t*> streams(plans.size());
  for( unsigned count = 0;count < streams.size();++count ){
    streams[count] = new cudaStream_t;
    HANDLE_ERROR(cudaStreamCreate(streams[count]));
  }
  

    HANDLE_ERROR(cudaMemcpyAsync(src_buffers[0],
				 forwarded_kernels[0].data(), 
				 reshaped_buffer_byte,
				 cudaMemcpyHostToDevice,
				 *streams[0]
				 ));
  

  for (int v = 0; v < _data.size(); ++v) {

    image_folds[v]->half_inplace<device_transform>(src_buffers[0],src_buffers[1],
						   streams[0], streams[1],
						   v+1 < _data.size() ? forwarded_kernels[v+1].data() : 0);
  }
    
  //clean-up
  for (unsigned count = 0;count < streams.size();++count){
    HANDLE_ERROR(cudaStreamSynchronize(*streams[count]));
    HANDLE_ERROR(cudaStreamDestroy(*streams[count]));
  }

  for (unsigned count = 0;count < src_buffers.size();++count){
    HANDLE_ERROR(cudaFree(src_buffers[count]));
  }

  for (unsigned count = 0;count < plans.size();++count){

    HANDLE_CUFFT_ERROR(cufftDestroy(*plans[count]));
    delete plans[count];
    plans[count] = 0;
  }


  for (int v = 0; v < _data.size(); ++v) {
    HANDLE_ERROR(cudaHostUnregister((void*)forwarded_kernels[v].data()));
    delete image_folds[v];
  }

}

typedef boost::chrono::high_resolution_clock::time_point tp_t;
typedef boost::chrono::milliseconds ms_t;
typedef boost::chrono::nanoseconds ns_t;

namespace po = boost::program_options;

// typedef boost::multi_array<float, 3, fftw_allocator<float> > fftw_image_stack;
// typedef std::vector<float, fftw_allocator<float> > aligned_float_vector;

int main(int argc, char* argv[]) {
  unsigned num_replicas = 8;
  bool verbose = false;
  // bool out_of_place = false;
  bool batched_plan = false;
  int device_id = -1;
  std::string cpu_name;
  


  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");
  
  //clang-format off
  desc.add_options()							//
    ("help,h", "produce help message")					//		
    ("verbose,v", "print lots of information in between")		//
    ("header-only,H", "print header of stats only")                   //
    									//
    ("stack_dimensions,s", 						//
     po::value<std::string>(&stack_dims)->default_value("512x512x64"), 	//
     "HxWxD of synthetic stacks to generate")				//
    									//
    ("repeats,r", 							//
     po::value<int>(&num_repeats)->default_value(10),			//
     "number of repetitions per measurement")				//
    									//
    ("num_replicas,n", 							//
     po::value<unsigned>(&num_replicas)->default_value(8), 			//
     "number of replicas to use for batched processing")		//
    									//
    ("device_id,d", 							//
     po::value<int>(&device_id)->default_value(-1),			//
     "cuda device to use")  					//
									//
    ("cpu_name,c", 							//
     po::value<std::string>(&cpu_name)->default_value("i7-3520M"),	//
     "cpu name to use in output")  					//
    ;									//
  //clang-format on
  
po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);

  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("header-only")) {
    print_header();
    return 0;
  }


  verbose = vm.count("verbose");
  // out_of_place = vm.count("out-of-place");
  batched_plan = vm.count("batched_plan");

  std::vector<unsigned> numeric_stack_dims;
  split<'x'>(stack_dims, numeric_stack_dims);

  if (verbose) {
    std::cout << "received " << numeric_stack_dims.size() << " dimensions: ";
    for (unsigned i = 0; i < numeric_stack_dims.size(); ++i) {
      std::cout << numeric_stack_dims[i] << " ";
    }
    std::cout << "\n";
  }

  if (numeric_stack_dims.size() != 3) {
    std::cerr << ">> " << numeric_stack_dims.size()
              << "-D data, not supported yet!\n";
    return 1;
  }

  std::vector<unsigned> reshaped(numeric_stack_dims);
  reshaped.back() = (reshaped.back() / 2 + 1) * 2;

  //////////////////////////////////////////////////////////////////////////////
  // set device flags
  if(device_id<0)
    device_id = selectDeviceWithHighestComputeCapability();
  
  HANDLE_ERROR(cudaSetDevice(device_id));
  unsigned long cufft_extra_space =
      cufft_3d_estimated_memory_consumption(numeric_stack_dims);
  unsigned long cufft_data_size = multiviewnative::gpu::cufft_r2c_memory(numeric_stack_dims);
  // unsigned long data_size_byte =
  //     std::accumulate(numeric_stack_dims.begin(), numeric_stack_dims.end(), 1u,
  //                     std::multiplies<unsigned long>()) *
  //     sizeof(float);
  unsigned long memory_available_on_device = getAvailableGMemOnCurrentDevice();

  float exp_mem_mb = (cufft_extra_space + cufft_data_size) / float(1 << 20);
  float av_mem_mb = memory_available_on_device / float(1 << 20);

  if (exp_mem_mb > av_mem_mb) {
    std::cerr << "not enough memory available on device, needed " << exp_mem_mb
              << " MB (data only: " << cufft_data_size / float(1 << 20)
              << " MB), available: " << av_mem_mb << " MB\n";
    return 1;
  } else {
    if (verbose)
      std::cout << "cufft memory estimate: needed " << exp_mem_mb
                << " MB (data only: " << cufft_data_size / float(1 << 20)
                << " MB), available: " << av_mem_mb << " MB\n";
  }



  multiviewnative::image_kernel_data raw(numeric_stack_dims);
  multiviewnative::image_kernel_data reference = raw;
  inplace_gpu_convolution(reference.stack_.data(),
			  &reference.stack_shape_[0],
			  reference.kernel_.data(),
			  &reference.kernel_shape_[0],
			  device_id);

  std::vector<multiviewnative::image_kernel_data> stacks(num_replicas,raw);


  if (verbose) {
    std::cout << "[config]\t"
              << "\n"
              << "num_replicas\t:\t" << num_replicas << "\nnumeric size\t:\t";
    std::copy(numeric_stack_dims.begin(), numeric_stack_dims.end(),
              std::ostream_iterator<unsigned>(std::cout, " "));

    std::cout << "\nfftw size\t:\t";
    std::copy(reshaped.begin(), reshaped.end(),
              std::ostream_iterator<unsigned>(std::cout, " "));
    std::cout << "\n";
  }


    stacks[0] = raw;

  //start measurement
  std::vector<ns_t> durations(num_repeats);

  ns_t time_ns = ns_t(0);
  tp_t start, end;
  for (int r = 0; r < num_repeats; ++r) {

    for ( multiviewnative::image_kernel_data& s : stacks ){
      s.stack_ = raw.stack_;
      s.kernel_ = raw.kernel_;
    }
    

    start = boost::chrono::high_resolution_clock::now();
    inplace_gpu_batched_fold(stacks);
    end = boost::chrono::high_resolution_clock::now();
    durations[r] = boost::chrono::duration_cast<ns_t>(end - start);

    time_ns += boost::chrono::duration_cast<ns_t>(end - start);
    if (verbose) {
      std::cout << r << "\t"
                << boost::chrono::duration_cast<ns_t>(durations[r]).count() /
	double(1e6) << " ms\n";
    }
  }

  bool data_valid = std::equal(reference.stack_.data(), reference.stack_.data() + reference.stack_.num_elements(),
			       stacks[0].stack_.data());
  

  std::string implementation_name = __FILE__;
  std::string comments = "global_plan";
  if(data_valid)
    comments += ",OK";
  else
    comments += ",NA";

  if(batched_plan)
    comments += ",batched";


  std::string device_name = get_cuda_device_name(device_id);
  std::replace(device_name.begin(), device_name.end(), ' ', '_');

  if(verbose)
    print_header();


  print_info(1,
	     implementation_name,
	     device_name,
	     num_repeats,
	     time_ns.count() / double(1e6),
	     numeric_stack_dims,
	     sizeof(float),
	     comments
	     );


  return 0;
}

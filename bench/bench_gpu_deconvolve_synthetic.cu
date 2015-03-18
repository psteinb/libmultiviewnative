#define __BENCH_GPU_DECONVOLVE_SYNTHETIC_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include "boost/program_options.hpp"
#include "synthetic_data.hpp"
//#include "cpu_nd_fft.hpp"
#include "multiviewnative.h"
#include "gpu_deconvolve_methods.cuh"

#include "logging.hpp"

#include <boost/chrono.hpp>
#include <boost/thread.hpp>


#include "gpu_convolve.cuh"
#include "padd_utils.h"
#include "gpu_nd_fft.cuh"
#include "cufft_utils.cuh"

#include "cuda_profiler.h"

typedef multiviewnative::zero_padd<multiviewnative::image_stack>
    wrap_around_padding;

typedef multiviewnative::no_padd<multiviewnative::image_stack>
    as_is_padding;

typedef multiviewnative::inplace_3d_transform_on_device<imageType>
    device_transform;

typedef multiviewnative::gpu_convolve<wrap_around_padding, imageType, unsigned>
    device_convolve;

//TODO:
//this is the convolution we wanna use in the end
typedef multiviewnative::gpu_convolve<as_is_padding, imageType, unsigned>
    target_convolve;

namespace mvn = multiviewnative;
// typedef mvn::no_padd<mvn::image_stack> stack_padding;
// typedef mvn::inplace_3d_transform_on_device<imageType>
//     device_transform;
// typedef mvn::gpu_convolve<stack_padding, imageType, unsigned>
//     device_convolve;


typedef boost::chrono::high_resolution_clock::time_point tp_t;
typedef boost::chrono::milliseconds ms_t;
typedef boost::chrono::nanoseconds ns_t;


namespace po = boost::program_options;

// typedef boost::multi_array<float, 3, fftw_allocator<float> > fftw_image_stack;
// typedef std::vector<float, fftw_allocator<float> > aligned_float_vector;

int main(int argc, char* argv[]) {
  unsigned num_views = 8;
  bool verbose = false;
  
  bool plan_many = false;
  int device_id = -1;
  


  int num_repeats = 5;
  std::string stack_dims = "";
  std::string mode = "";

  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()                                                      //
      ("help,h", "produce help message")                                  //
      ("verbose,v", "print lots of information in between")               //
      ("header-only,H", "print header of stats only")                     //
                                                                          //
      ("stack_dimensions,s",                                              //
       po::value<std::string>(&stack_dims)->default_value("64x64x64"),  //
       "HxWxD of synthetic stacks to generate")                           //
                                                                          //
      ("repeats,r",                                                       //
       po::value<int>(&num_repeats)->default_value(10),                   //
       "number of repetitions per measurement")                           //
                                                                          //
      ("num_views,n",                                                  //
       po::value<unsigned>(&num_views)->default_value(6),              //
       "number of replicas to use for batched processing")                //
                                                                          //
      ("device_id,d",                                                     //
       po::value<int>(&device_id)->default_value(-1),                     //
       "cuda device to use")                                              //
      ("mode,t",                                                     //
       po::value<std::string>(&mode)->default_value("all_on_device"),                     //
       "choost deconvolution mode (all_on_device,interleaved)")                                              //

      ;                                                                   //
  // clang-format on

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

  if( ! (mode == "all_on_device" || mode == "interleaved" ) ){
    std::cerr << "unknown mode : " << mode << " Exiting ...\n";
    return 1;
  }

  verbose = vm.count("verbose");
  // out_of_place = vm.count("out-of-place");
  plan_many = vm.count("plan_many");

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

  float exp_mem_mb = ((cufft_extra_space + cufft_data_size)*num_views*4) / float(1 << 20);
  float av_mem_mb = memory_available_on_device / float(1 << 20);

  if( mode == "all_on_device" && exp_mem_mb > av_mem_mb) {
    std::cerr << "[all_on_device] not enough memory available on device, needed " << exp_mem_mb
              << " MB (data only: " << cufft_data_size / float(1 << 20)
              << " MB), available: " << av_mem_mb << " MB\n";
    return 1;
  } 


  multiviewnative::multiview_data running(numeric_stack_dims, num_views);

  ns_t time_ns = ns_t(0);
  tp_t start, end;

  workspace input;
    input.data_ = 0;
    
    running.fill_workspace(input);
    input.num_iterations_ = num_repeats;
    input.lambda_ = .006;
    input.minValue_ = .001;
        
    multiviewnative::shape_t shape(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3);
    multiviewnative::image_stack start_psi(shape);
    
    cudaProfilerStart();
    start = boost::chrono::high_resolution_clock::now();

    if(mode == "interleaved")
      inplace_gpu_deconvolve_iteration_interleaved<as_is_padding, 
						   target_convolve,
						   device_transform>(start_psi.data(), input, device_id);
    else
      inplace_gpu_deconvolve_iteration_all_on_device<wrap_around_padding, device_transform>(start_psi.data(), input, device_id);
    
    end = boost::chrono::high_resolution_clock::now();
    cudaProfilerEnd();
    
    time_ns += boost::chrono::duration_cast<ns_t>(end - start);
      

      

  std::string implementation_name = __FILE__;
  std::stringstream comments("");
  comments << mode;
  comments << ",NA," << "nstacks=" << num_views*4;
  

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
	     comments.str()
	     );

  // tear-down
  delete[] input.data_;

  return 0;
}

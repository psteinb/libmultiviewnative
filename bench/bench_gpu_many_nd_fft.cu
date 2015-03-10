#define __BENCH_GPU_DECONVOLVE_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>

#include "boost/program_options.hpp"
#include "synthetic_data.hpp"

#include "cuda_profiler_api.h"
#include "gpu_nd_fft.cuh"
#include "logging.hpp"

#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {

  bool verbose = false;
  bool out_of_place = false;
  cufftHandle* global_plan = 0;
  std::string tx_mode = "sync";

  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()                                                          //
      ("help,h", "produce help message")                                      //
      ("verbose,v", "print lots of information in between")                   //
      ("out-of-place,o", "perform out-of-place transforms")                   //
      ("stack_dimensions,s",                                                  //
       po::value<std::string>(&stack_dims)->default_value("512x512x64"),      //
       "DxHxW of synthetic stacks to generate")                               //
      ("repeats,r", po::value<int>(&num_repeats)->default_value(10),          //
       "number of repetitions per measurement")                               //
      ("tx_mode,t", po::value<std::string>(&tx_mode)->default_value("sync"),  //
       "transfer mode of data\n(possible values: sync, async, man, evts)")    //
      ;                                                                       //
  // clang-format on

  po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);

  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  verbose = vm.count("verbose");
  out_of_place = vm.count("out-of-place");

  for (char& c : tx_mode) c = std::tolower(c);

  if (!(tx_mode == "sync" || tx_mode == "async" || tx_mode == "man" ||
        tx_mode == "evts")) {
    std::cout << "transfer mode: " << tx_mode << " not supported";
    return 1;
  }

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

  //////////////////////////////////////////////////////////////////////////////
  // set device flags
  int device_id = selectDeviceWithHighestComputeCapability();
  if (tx_mode == "man") {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    if (!prop.canMapHostMemory) {
      std::cerr << "device " << device_id << " cannot map host memory";
      return 1;
    } else {
      HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
    }
  }
  //////////////////////////////////////////////////////////////////////////////
  // estimate memory

  HANDLE_ERROR(cudaSetDevice(device_id));
  unsigned long cufft_extra_space =
      cufft_3d_estimated_memory_consumption(numeric_stack_dims);
  unsigned long cufft_data_size = cufft_r2c_memory(numeric_stack_dims);
  unsigned long data_size_byte =
      std::accumulate(numeric_stack_dims.begin(), numeric_stack_dims.end(), 1u,
                      std::multiplies<unsigned long>()) *
      sizeof(float);
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

  //////////////////////////////////////////////////////////////////////////////
  // PLAN

  global_plan = new cufftHandle;

  HANDLE_CUFFT_ERROR(cufftPlan3d(global_plan, (int)numeric_stack_dims[0],
                                 (int)numeric_stack_dims[1],
                                 (int)numeric_stack_dims[2], CUFFT_R2C));

  HANDLE_CUFFT_ERROR(
      cufftSetCompatibilityMode(*global_plan, CUFFT_COMPATIBILITY_NATIVE));

  //////////////////////////////////////////////////////////////////////////////
  // generate data
  multiviewnative::image_kernel_data data(numeric_stack_dims);
  std::random_shuffle(data.stack_.data(),
                      data.stack_.data() + data.stack_.num_elements());
  

  static const int num_stacks = 8;
  std::vector<multiviewnative::image_stack> stacks(num_stacks, data.stack_);

  if (verbose) {
    std::cout << "[config]\t" << ((out_of_place) ? "out-of-place" : "inplace")
              << " "
              << "global_plan "
	      << tx_mode << " "
              << "\n";
  }

  std::vector<cpu_times> durations(num_repeats);

  double time_ms = 0.f;
  float* d_dest_buffer = 0;
  const unsigned fft_size_in_byte_ = cufft_r2c_memory(numeric_stack_dims);
  std::vector<int> fft_reshaped = cufft_r2c_shape(numeric_stack_dims);
  

  if (out_of_place)
    HANDLE_ERROR(cudaMalloc((void**)&(d_dest_buffer), fft_size_in_byte_));
  else {
    for (multiviewnative::image_stack& stack : stacks)
      stack.resize(fft_reshaped);

  }
  if(verbose){
    std::cout << "[fftshape]\t" 
	      << fft_reshaped[0] << "x"
	      << fft_reshaped[1] << "x"
      	      << fft_reshaped[2] << "\n[stackshape] "
	      << stacks[0].shape()[0] << "x"
	      << stacks[0].shape()[1] << "x"
      	      << stacks[0].shape()[2] << "\n";
  }

  float* d_src_buffer = 0;

  if (out_of_place)
    HANDLE_ERROR(cudaMalloc((void**)&(d_src_buffer), data_size_byte));
  else
    HANDLE_ERROR(cudaMalloc((void**)&(d_src_buffer), fft_size_in_byte_));

  // warm-up
  multiviewnative::image_stack reference = stacks[0];
  multiviewnative::image_stack raw = stacks[0];
  fft_incl_transfer_excl_alloc(reference, 
			       d_src_buffer,
                               out_of_place ? d_dest_buffer : 0, 
			       global_plan);

  //////////////////////////////////////////////////////////////////////////////
  // do not include allocations in time measurement
  if (tx_mode == "sync") {

    cudaProfilerStart();
    for (int r = 0; r < num_repeats; ++r) {
      std::fill(stacks.begin(), stacks.end(), raw);
      cpu_timer timer;
      batched_fft_synced(stacks, d_src_buffer, out_of_place ? d_dest_buffer : 0,
                         global_plan);
      durations[r] = timer.elapsed();

      time_ms += double(durations[r].system + durations[r].user) / 1e6;
      if (verbose) {
        std::cout << "synced  " << r << "\t"
                  << double(durations[r].system + durations[r].user) / 1e6
                  << " ms\n";
      }
      
    }
    cudaProfilerStop();


  }

  if (tx_mode == "man") {

    cudaProfilerStart();
    for (int r = 0; r < num_repeats; ++r) {
      std::fill(stacks.begin(), stacks.end(), raw);
      cpu_timer timer;
      batched_fft_managed(stacks, d_src_buffer,
                          out_of_place ? d_dest_buffer : 0, global_plan);
      durations[r] = timer.elapsed();

      time_ms += double(durations[r].system + durations[r].user) / 1e6;
      if (verbose) {
        std::cout << "managed " << r << "\t"
                  << double(durations[r].system + durations[r].user) / 1e6
                  << " ms\n";
      }
      
    }
    cudaProfilerStop();
  }

  if (tx_mode == "async") {

    cudaProfilerStart();
    for (int r = 0; r < num_repeats; ++r) {
      std::fill(stacks.begin(), stacks.end(), raw);
      cpu_timer timer;
      batched_fft_async(stacks, d_src_buffer,
                          out_of_place ? d_dest_buffer : 0, global_plan);
      durations[r] = timer.elapsed();

      time_ms += double(durations[r].system + durations[r].user) / 1e6;
      if (verbose) {
        std::cout << "async " << r << "\t"
                  << double(durations[r].system + durations[r].user) / 1e6
                  << " ms\n";
      }
      
    }
    cudaProfilerStop();
  }
  //check if all transforms worked as expected
  double ref_sum = std::accumulate(reference.data(), reference.data() + reference.num_elements()/4, 0.);
  double running_sum = 0.;
  unsigned matching_results = 0;
  int count = 0;
  for ( const multiviewnative::image_stack& stack : stacks ){
    running_sum = std::accumulate(stack.data(), stack.data() + stack.num_elements()/4, 0.);
    if(std::abs(running_sum - ref_sum)/ref_sum < 1e-2)
      matching_results++;
    if(verbose)
      std::cout << "checksum " << count++ << " " << running_sum << " (ref: " << ref_sum << ")\n";
  }

  if(matching_results!=stacks.size()){
    std::cerr << "ERROR! results do not validate, only " << matching_results << "/"<< stacks.size()<<" image stacks comply with reference\n";
  }

  HANDLE_ERROR(cudaFree(d_src_buffer));

  if (out_of_place) HANDLE_ERROR(cudaFree(d_dest_buffer));

  HANDLE_CUFFT_ERROR(cufftDestroy(*global_plan));
  delete global_plan;

  std::string device_name = get_cuda_device_name(device_id);
  std::replace(device_name.begin(), device_name.end(), ' ', '_');

  if (verbose) print_header();

  std::stringstream comments;
  comments << tx_mode << "," << ((out_of_place) ? "out-of-place" : "inplace")
           << ","
           << "global_plan";

  print_info(1, __FILE__, device_name, num_repeats, time_ms, numeric_stack_dims,
             sizeof(float), comments.str());

  return 0;
}

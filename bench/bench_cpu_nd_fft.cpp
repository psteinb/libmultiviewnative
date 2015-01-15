#define __BENCH_GPU_DECONVOLVE_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include "boost/program_options.hpp" 
#include "synthetic_data.hpp"
#include "cpu_nd_fft.hpp"
#include "fftw_interface.h"

#include <boost/chrono.hpp>
#include <boost/thread.hpp>

// // #include <boost/timer/timer.hpp>
// using boost::timer::cpu_timer;
// using boost::timer::cpu_times;
// using boost::timer::nanosecond_type;

typedef boost::chrono::high_resolution_clock::time_point tp_t;
typedef boost::chrono::milliseconds ms_t;
typedef boost::chrono::nanoseconds ns_t;

namespace po = boost::program_options;

typedef boost::multi_array<float,3, fftw_allocator<float> >    fftw_image_stack;
//typedef boost::multi_array<float,3 >    fftw_image_stack;

int main(int argc, char *argv[])
{

  bool verbose = false;
  bool reuse_global_plan = false;
  bool out_of_place = false;
  bool use_global_plan = false;
  int num_threads = 0;		
  
  fftw_api::plan_type* global_plan = 0;
  
  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("verbose,v", "print lots of information in between")
    ("stack_dimensions,s", po::value<std::string>(&stack_dims)->default_value("512x512x64"), "HxWxD of synthetic stacks to generate")
    ("global_plan,g", "use a global plan, rather than creating a plan everytime a transformation is performed" )
    ("reuse_global_plan,a", "use a global plan, and reuse it for all transforms" )
    ("out-of-place,o", "perform out-of-place transforms" )
    ("repeats,r", po::value<int>(&num_repeats)->default_value(10), "number of repetitions per measurement")
    ("num_threads,t", po::value<int>(&num_threads)->default_value(1), "number of threads to use") // 
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
  out_of_place = vm.count("out-of-place");
  use_global_plan = vm.count("global_plan");
  reuse_global_plan = vm.count("reuse_global_plan");

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
  
  unsigned long data_size_byte = std::accumulate(numeric_stack_dims.begin(), numeric_stack_dims.end(), 1u, std::multiplies<unsigned long>())*sizeof(float);
  unsigned long memory_available = data_size_byte;//later

  float exp_mem_mb = (data_size_byte)/float(1 << 20);
  float av_mem_mb = memory_available/float(1 << 20);

  if(exp_mem_mb>av_mem_mb){
    std::cerr << "not enough memory available on device, needed " << exp_mem_mb 
	      <<" MB), available: " << av_mem_mb << " MB\n";
    return 1;
  } else {
    if(verbose)
      std::cout << "[NOT IMPLEMENTED YET] memory estimate: needed " << exp_mem_mb 
		<<" MB), available: " << av_mem_mb << " MB\n";
  }

  


  std::vector<unsigned> reshaped(numeric_stack_dims);
  fftw_r2c_reshape(reshaped);

  bool shapes_differ = !std::equal(numeric_stack_dims.begin(), 
				   numeric_stack_dims.end(), 
				   reshaped.begin());

  fftw_image_stack aligned_input(shapes_differ ? reshaped : numeric_stack_dims);
  for(unsigned long i = 0;i < aligned_input.num_elements();++i)
    aligned_input.data()[i] = float(i);
  
  std::random_shuffle(aligned_input.data(),aligned_input.data() + aligned_input.num_elements());

  float* d_dest_buffer = 0; 
  fftw_image_stack* aligned_output = 0;

  if(out_of_place){
    aligned_output = new fftw_image_stack(reshaped);
    d_dest_buffer = aligned_output->data();
  }
  else {
    bool too_small = false;
    for (int i = 0; i < 3; ++i)
      {
	if(numeric_stack_dims[i] < reshaped[i]){
	  too_small = true;
	  break;
	}
      }

    if(too_small){
      aligned_input.resize(reshaped);
      if(verbose){
	std::cout << "adjusting input data size\t";
	for (unsigned i = 0; i < reshaped.size(); ++i)
	  {
	    std::cout << reshaped[i] << " ";
	  }
	std::cout << "\n";
      }
    }
  }


  if(verbose){
    std::cout << "[config]\t" 
	      << "excl_alloc" << " " 
	      << "excl_tx" << " " 
	      << ( (out_of_place) ? "out-of-place" : "inplace") << " " 
      	      << ( (use_global_plan) ? "global plans" : "local plans") << " " 
	      << "\n";
    std::copy(numeric_stack_dims.begin(),
	      numeric_stack_dims.end(),
	      std::ostream_iterator<unsigned>(std::cout, " "));
    std::cout << "\n";
  }
  
  static const int max_threads = boost::thread::hardware_concurrency();
  if(num_threads>1 && num_threads<=max_threads){
    fftw_api::init_threads();
    fftw_api::plan_with_threads(num_threads);
    if(verbose)
      std::cout << "planning with " << num_threads << " threads\n";
  }


  tp_t start, end;
  if(use_global_plan){
    global_plan = new fftw_api::plan_type ;
    start = boost::chrono::high_resolution_clock::now();
    *global_plan = fftw_api::dft_r2c_3d(numeric_stack_dims[0], numeric_stack_dims[1], numeric_stack_dims[2],
					(fftw_api::real_type*)aligned_input.data(), 
					(fftw_api::complex_type*)(out_of_place ?  aligned_output->data() : aligned_input.data() ),
					reuse_global_plan ? FFTW_MEASURE : FFTW_ESTIMATE);
    end = boost::chrono::high_resolution_clock::now();
    if(verbose)
      std::cout << "creating plan took ("<< (reuse_global_plan ? "FFTW_MEASURE" : "FFTW_ESTIMATE")
		<<") " << boost::chrono::duration_cast<ms_t>(end - start) << "\n";
  }

   
  std::vector<ns_t> durations(num_repeats);
  
  
  
  ns_t time_ns = ns_t(0);
  
    for(int r = 0;r<num_repeats;++r){
      start = boost::chrono::high_resolution_clock::now();
      
      if(!reuse_global_plan){
	if(verbose) std::cout << "[process_plan]\t";
	st_fftw(numeric_stack_dims,
		aligned_input.data(),
		out_of_place ? d_dest_buffer : 0 ,
		use_global_plan ? global_plan : 0);
      }
      else{
	  if(verbose) std::cout << "[reuse_plan]\t";
	  reuse_fftw_plan(numeric_stack_dims,
			aligned_input.data(),
			out_of_place ? d_dest_buffer : 0,
			global_plan);


      }

      end = boost::chrono::high_resolution_clock::now();
      durations[r] = boost::chrono::duration_cast<ns_t>(end - start);

      time_ns += boost::chrono::duration_cast<ns_t>(end - start);
      if(verbose){
	std::cout << r << "\t" << boost::chrono::duration_cast<ns_t>(durations[r]).count()/double(1e6) << " ms\n";

      }
    }

  if(out_of_place){
    delete aligned_output;
    //d_dest_buffer = 0;
  }

  if(use_global_plan){
    fftw_api::destroy_plan(*global_plan);
    delete global_plan;
  }

  std::string device_name = "CPU";
  
  std::cout << num_threads << "x" << device_name << " "
	    << "excl_alloc" << " " 
	    << "incl_tx" << " " 
	    << ( (out_of_place) ? "out-of-place" : "inplace") << " " 
	    << num_repeats <<" " 
	    << time_ns.count()/double(1e6) << " " 
	    << stack_dims << " " 
	    << data_size_byte/float(1 << 20) << " " 
	    << exp_mem_mb
	    << "\n";

  if(num_threads>1 && num_threads<=max_threads)
    fftw_api::cleanup_threads();

  return 0;
}

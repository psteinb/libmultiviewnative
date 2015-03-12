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

#include "logging.hpp"

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

typedef boost::multi_array<float, 3, fftw_allocator<float> > fftw_image_stack;
typedef std::vector<float, fftw_allocator<float> > aligned_float_vector;

int main(int argc, char* argv[]) {
  unsigned num_replicas = 8;
  bool verbose = false;
  // bool out_of_place = false;
  bool batched_plan = false;
  int num_threads = 0;
  std::string cpu_name;
  
  fftw_api::plan_type* global_plan = 0;

  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");
  
  //clang-format off
  desc.add_options()							//
    ("help,h", "produce help message")					//		
    ("verbose,v", "print lots of information in between")		//
    ("batched_plan,b","use a batched plan")				//
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
    ("num_threads,t", 							//
     po::value<int>(&num_threads)->default_value(1),			//
     "number of threads to use")  					//
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

  unsigned long data_size_byte =
      std::accumulate(numeric_stack_dims.begin(), numeric_stack_dims.end(), 1u,
                      std::multiplies<unsigned long>()) *
      sizeof(float);
  unsigned long memory_available = data_size_byte;  // later

  float exp_mem_mb = (data_size_byte) / float(1 << 20);
  exp_mem_mb *= num_replicas;

  float av_mem_mb = memory_available / float(1 << 20);

  // if(exp_mem_mb>av_mem_mb){
  //   std::cerr << "not enough memory available on device, needed " <<
  // exp_mem_mb
  // 	      <<" MB), available: " << av_mem_mb << " MB\n";
  //   return 1;
  // } else {
  if (verbose)
    std::cout << "[NOT IMPLEMENTED YET] memory estimate: needed " << exp_mem_mb
              << " MB), available: " << av_mem_mb << " MB\n";
  // }

  std::vector<unsigned> reshaped(numeric_stack_dims);
  reshaped.back() = (reshaped.back() / 2 + 1) * 2;

  fftw_image_stack aligned_input(reshaped);
  
  const unsigned long total_pixel_count =
      num_replicas * aligned_input.num_elements();

  if (verbose)
    std::cout << "creating vector of " << total_pixel_count
              << " float elements\n";
  aligned_float_vector many_inputs_flat(total_pixel_count);

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

  static const int max_threads = boost::thread::hardware_concurrency();
  if (num_threads > 1 && num_threads <= max_threads) {
    fftw_api::init_threads();
    fftw_api::plan_with_threads(num_threads);
    if (verbose) std::cout << "planning with " << num_threads << " threads\n";
  }

  tp_t start, end;
  if (!batched_plan) {  //
    global_plan = new fftw_api::plan_type;
    start = boost::chrono::high_resolution_clock::now();
    *global_plan = fftw_api::dft_r2c_3d(
        numeric_stack_dims[0], numeric_stack_dims[1], numeric_stack_dims[2],
        (fftw_api::real_type*)aligned_input.data(),
        (fftw_api::complex_type*)(aligned_input.data()), FFTW_MEASURE);
    end = boost::chrono::high_resolution_clock::now();
    if (verbose)
      std::cout << "creating standard plan took (FFTW_MEASURE) "
                << boost::chrono::duration_cast<ms_t>(end - start) << "\n";
  } else {
    global_plan = new fftw_api::plan_type;
    start = boost::chrono::high_resolution_clock::now();
    *global_plan = fftw_api::dft_r2c_many(              //
        numeric_stack_dims.size(),                      // int rank
        (int *)&numeric_stack_dims[0],                  // const int* n
        num_replicas,                                   // int howmany
        (fftw_api::real_type *)&many_inputs_flat[0],    // float* in
        (int *)0,                                       // const int* inembed
        1,                                              // int istride
        (int)aligned_input.num_elements(),              // int idist
        (fftw_api::complex_type *)&many_inputs_flat[0], // fftw_complex* out
        (int *)0,                                       // const int* oembed
        1,                                              // int ostride,
        (int)aligned_input.num_elements() / 2,          // int odist
        FFTW_MEASURE);                                  //
    end = boost::chrono::high_resolution_clock::now();
    if (verbose)
      std::cout << "creating batched plan took (FFTW_MEASURE) "
                << boost::chrono::duration_cast<ms_t>(end - start) << "\n";
  }

  //fill data
  for ( unsigned i = 0;i < aligned_input.num_elements();++i )
    {
      aligned_input.data()[i] = float(i);
    }
  fftw_image_stack original = aligned_input;
  for ( unsigned i = 0;i < num_replicas;++i){
    std::copy(aligned_input.data(), aligned_input.data() + aligned_input.num_elements(),
	      &many_inputs_flat[0] + (i*aligned_input.num_elements()));
  }

  //start measurement
  std::vector<ns_t> durations(num_repeats);

  ns_t time_ns = ns_t(0);

  for (int r = 0; r < num_repeats; ++r) {

    for ( unsigned i = 0;i < num_replicas;++i){
      std::copy(original.data(), original.data() + original.num_elements(),
		&many_inputs_flat[0] + (i*original.num_elements()));
    }

    start = boost::chrono::high_resolution_clock::now();
    if (!batched_plan) {
      for (unsigned long count = 0; count < total_pixel_count;
           count += aligned_input.num_elements()) {
        reuse_fftw_plan(numeric_stack_dims, &many_inputs_flat[count], 0,
                        global_plan);
      }
    } else {
      fftw_api::execute_plan(*global_plan);
    }
    end = boost::chrono::high_resolution_clock::now();
    durations[r] = boost::chrono::duration_cast<ns_t>(end - start);

    time_ns += boost::chrono::duration_cast<ns_t>(end - start);
    if (verbose) {
      std::cout << r << "\t"
                << boost::chrono::duration_cast<ns_t>(durations[r]).count() /
                       double(1e6) << " ms\n";
    }
  }

  bool data_valid = std::equal(&many_inputs_flat[0], &many_inputs_flat[0] + original.num_elements(),
			       &many_inputs_flat[(num_replicas-1)*original.num_elements()]);
  
  fftw_api::destroy_plan(*global_plan);
  delete global_plan;

  std::string implementation_name = __FILE__;
  std::string comments = "global_plan";
  if(data_valid)
    comments += ",OK";
  else
    comments += ",NA";

  if(batched_plan)
    comments += ",batched";

  if(verbose)
    print_header();

  print_info(num_threads,
	     implementation_name,
	     cpu_name,
	     num_repeats,
	     time_ns.count() / double(1e6),
	     numeric_stack_dims,
	     sizeof(float),
	     comments
	     );

  if (num_threads > 1 && num_threads <= max_threads)
    fftw_api::cleanup_threads();

  return 0;
}

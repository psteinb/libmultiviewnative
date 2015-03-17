#define __BENCH_CPU_DECONVOLVE_SYNTHETIC_CPP__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include "boost/program_options.hpp"
#include "synthetic_data.hpp"
//#include "cpu_nd_fft.hpp"
#include "multiviewnative.h"

#include "logging.hpp"

#include <boost/chrono.hpp>
#include <boost/thread.hpp>


typedef std::vector<unsigned> shape_t;

namespace mvn = multiviewnative;


typedef boost::chrono::high_resolution_clock::time_point tp_t;
typedef boost::chrono::milliseconds ms_t;
typedef boost::chrono::nanoseconds ns_t;


namespace po = boost::program_options;

// typedef boost::multi_array<float, 3, fftw_allocator<float> > fftw_image_stack;
// typedef std::vector<float, fftw_allocator<float> > aligned_float_vector;

int main(int argc, char* argv[]) {
  unsigned num_views = 8;
  bool verbose = false;
  
  int num_threads = -1;
  


  int num_repeats = 5;
  std::string stack_dims = "";
  std::string cpu_name = "";

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
      ("cpu_name,c",                                                     //
       po::value<std::string>(&cpu_name)->default_value("local-cpu"),                     //
       "cpu name to use")                                              //
      ("threads,t",                                                     //
       po::value<int>(&num_threads)->default_value(1),                     //
       "how many threads to use")                                              //

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

  static const int max_threads = boost::thread::hardware_concurrency();
  if(num_threads > max_threads || num_threads < 1)
    num_threads  = max_threads;
  
  verbose = vm.count("verbose");
  

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

  multiviewnative::multiview_data running(numeric_stack_dims, num_views);

  ns_t time_ns = ns_t(0);
  tp_t start, end;

  workspace input;
    input.data_ = 0;
    
    running.fill_workspace(input);
    input.num_iterations_ = num_repeats;
    input.lambda_ = .006;
    input.minValue_ = .001;
        
    shape_t shape(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3);
    multiviewnative::image_stack start_psi(shape);
  
    start = boost::chrono::high_resolution_clock::now();

    inplace_cpu_deconvolve(start_psi.data(), input, num_threads);
    
    end = boost::chrono::high_resolution_clock::now();
    
    time_ns += boost::chrono::duration_cast<ns_t>(end - start);
      

      

  std::string implementation_name = __FILE__;
  std::stringstream comments("");
  comments << "nstacks=" << num_views*4;
  

  if(verbose)
    print_header();

  print_info(num_threads,
	     __FILE__,
	     cpu_name,
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

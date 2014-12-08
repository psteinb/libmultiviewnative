#define __BENCH_GPU_DECONVOLVE_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#include "boost/program_options.hpp" 
#include "boost/regex.hpp"

#include "test_fixtures.hpp"
#include "image_stack_utils.h"
#include "multiviewnative.h"
#include "cpu_kernels.h"

#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;

namespace po = boost::program_options;

void split(const std::string& _s, std::vector<int>& _tgt){
  boost::regex re("x");
  boost::sregex_token_iterator i(_s.begin(), _s.end(), re, -1);
  boost::sregex_token_iterator j;

  unsigned exp_numbers = std::count(_s.begin(), _s.end(), 'x') + 1;
  if(_tgt.size()< exp_numbers)
    _tgt.resize(exp_numbers);
  
  unsigned count = 0;
  std::string substr;
  while(i != j)
    {
      substr = *i++;
      std::istringstream istr(substr);
      try{
	istr >> _tgt[count];
      }
      catch(...){
	std::cerr << "unable to convert" << substr.c_str() << " to int\n";
      }
      count++;
    }

      
}

#ifndef CUDA_HOST_PREFIX
#ifdef __CUDACC__
#define CUDA_HOST_PREFIX inline __host__
#else
#define CUDA_HOST_PREFIX 
#endif
#endif

struct simulated_data {

  std::vector<multiviewnative::image_stack> views_;
  std::vector<multiviewnative::image_stack> kernel1_;
  std::vector<multiviewnative::image_stack> kernel2_;
  std::vector<multiviewnative::image_stack> weights_;
    
  std::vector<int> stack_dims_;
  std::vector<int> kernel1_dims_;
  std::vector<int> kernel2_dims_;

  CUDA_HOST_PREFIX simulated_data(const std::vector<int>& stack_dims, int _n_views = 6):
    views_(_n_views),
    kernel1_(_n_views),
    kernel2_(_n_views),
    weights_(_n_views),
    stack_dims_(stack_dims),
    kernel1_dims_(stack_dims),
    kernel2_dims_(stack_dims)
  {
      
      
    for(unsigned d = 0;d < stack_dims.size(); ++d){
      kernel1_dims_[d] *= .05*(d+1);
      kernel2_dims_[d] *= .02*(d+1);
    }

    unsigned int num_elements = std::accumulate(stack_dims.begin(), stack_dims.end(), 1, std::multiplies<int>());

    for(int i = 0; i < _n_views; ++i){
	
	
      views_[i].resize(boost::extents[stack_dims_[0]][stack_dims_[1]][stack_dims_[2]]);
      weights_[i].resize(boost::extents[stack_dims_[0]][stack_dims_[1]][stack_dims_[2]]);
      std::fill(weights_[i].data(), weights_[i].data() + num_elements, 1.f);
      std::fill(views_[i].data(), views_[i].data() + num_elements, 16.f + 4.*i);
	
      kernel1_[i].resize(boost::extents[kernel1_dims_[0]][kernel1_dims_[1]][kernel1_dims_[2]]);
      kernel1_[i][kernel1_dims_[0]/2][kernel1_dims_[1]/2][kernel1_dims_[2]/2] = i+1;
      kernel2_[i].resize(boost::extents[kernel2_dims_[0]][kernel2_dims_[1]][kernel2_dims_[2]]);
      kernel2_[i][kernel2_dims_[0]/2][kernel2_dims_[1]/2][kernel2_dims_[2]/2] = i+2;

    }

  }

  void info() {
    int nviews = views_.size();
    std::cout << "[simulated_data] " 
	      << nviews << "x views/weights "<< views_[0].shape()[0] << "x"  << views_[0].shape()[1] << "x"  << views_[0].shape()[2] 
	      << " kernel1 "<< kernel1_[0].shape()[0] << "x"  << kernel1_[0].shape()[1] << "x"  << kernel1_[0].shape()[2] 
	      << " kernel2 "<< kernel2_[0].shape()[0] << "x"  << kernel2_[0].shape()[1] << "x"  << kernel2_[0].shape()[2] 
	      << "\n";

  }
};

void default_inplace_gpu_deconvolve(float* _psi, workspace* _ws, int device = -1){


  unsigned long stack_size = _ws->data_[0].image_dims_[0]*_ws->data_[0].image_dims_[1]*_ws->data_[0].image_dims_[2];
  std::vector<float> integral(stack_size);

  if(device < 0)
    device = selectDeviceWithHighestComputeCapability();

  for(int it = 0;it<_ws->num_iterations_;++it){

    for(int v = 0;v < _ws->num_views_;++v){
      
      std::copy(_psi, _psi + stack_size, integral.begin());
      
      convolution3DfftCUDAInPlace(&integral[0], _ws->data_[v].image_dims_,
				  _ws->data_[v].kernel1_,_ws->data_[v].kernel1_dims_,device);
      
      //serial
      computeQuotient(_ws->data_[v].image_, &integral[0], stack_size);

      convolution3DfftCUDAInPlace(&integral[0], _ws->data_[v].image_dims_,
				  _ws->data_[v].kernel2_,_ws->data_[v].kernel2_dims_,device);
      //serial
      serial_regularized_final_values(_psi, &integral[0], _ws->data_[v].weights_, 
				      stack_size,
				      _ws->lambda_ ,
				      _ws->minValue_ );
    }
    
  }

}


int main(int argc, char *argv[])
{

  bool verbose = false;
  bool use_default = false;
  int num_views = 6;
  int num_iterations = 5;
  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("verbose,v", "print lots of information in between")
    ("use_default,d", "use implementation used in the original SPIM_Registration plugin ")
    ("stack_dimensions,s", po::value<std::string>(&stack_dims)->default_value("512x512x64"), "HxWxD of synthetic stacks to generate")
    ("views,n", po::value<int>(&num_views)->default_value(6), "number of views to generate")
    ("iterations,i", po::value<int>(&num_iterations)->default_value(5), "location of kernel image")
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
  use_default = vm.count("use_default");

  std::vector<int> numeric_stack_dims(3);
  numeric_stack_dims[0] = 512;
  numeric_stack_dims[1] = 512;
  numeric_stack_dims[2] = 64;
  
  split(stack_dims,numeric_stack_dims);

  simulated_data data(numeric_stack_dims, num_views);
  data.info();

  workspace input;
  input.data_ = new view_data[num_views];
  input.num_iterations_ = num_iterations;
  input.num_views_ = num_views;
  input.minValue_ = 1e-5;
  input.lambda_ = 6e-2;

  for(int v = 0;v < num_views; ++v){
    input.data_[v].image_ = data.views_[v].data();
    input.data_[v].weights_ = data.weights_[v].data();
    input.data_[v].kernel1_ = data.kernel1_[v].data();
    input.data_[v].kernel2_ = data.kernel2_[v].data();

    input.data_[v].image_dims_   = &data.stack_dims_[0];
    input.data_[v].weights_dims_ = &data.stack_dims_[0];
    input.data_[v].kernel1_dims_ = &data.kernel1_dims_[0];
    input.data_[v].kernel2_dims_ = &data.kernel2_dims_[0];
    
  }
  
  std::vector<cpu_times> durations(num_repeats);
  std::vector<float> psi(data.views_[0].data(), data.views_[0].data() + data.views_[0].num_elements());
  double time_ms = 0.f;
  
  for(int r = 0;r<num_repeats;++r){
    
    
    if(use_default){
      if(verbose)
	std::cout << "[default inplace_gpu_deconvolve] repeat " << r << "/" << num_repeats;
      cpu_timer timer;
      default_inplace_gpu_deconvolve(&psi[0], &input, -1);
      durations[r] = timer.elapsed();
    }
    else{
      if(verbose)
	std::cout << "[inplace_gpu_deconvolve] repeat " << r << "/" << num_repeats;
    cpu_timer timer;
    inplace_gpu_deconvolve(&psi[0], input, -1);
    durations[r] = timer.elapsed();
    }

    time_ms += double(durations[r].system + durations[r].user)/1e6;
    if(verbose){
      std::cout << " took " << double(durations[r].system + durations[r].user)/1e6 << " ms\n";
    }
  }

  std::cout << "[bench_gpu_deconvolve] " << time_ms << " ms\n";

  delete [] input.data_;

  return 0;
}

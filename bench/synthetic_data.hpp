#ifndef _SYNTHETIC_DATA_H_
#define _SYNTHETIC_DATA_H_

#include <sstream>
#include <vector>
#include <string>

#include "image_stack_utils.h"

template<const char token, typename T>
void split(const std::string& _s, std::vector<T>& _tgt){

  unsigned n_dims = std::count(_s.begin(), _s.end(), token) + 1;
  _tgt.resize(n_dims);

  size_t found_ = 0;
  unsigned count = 0;
  while(found_ != std::string::npos){
    size_t next_ = _s.find(token,found_);
    std::string temp = _s.substr(found_, next_-found_);
    std::istringstream istr(temp);
    try{
      istr >> _tgt[count++];
    }
    catch(...){
      std::cerr << "unable to convert" << temp.c_str() << " to int\n";
    }

    if(next_ != std::string::npos)
      found_ = next_+1;
    else
      found_ = std::string::npos;

  }

 
      
}

#ifndef CUDA_HOST_PREFIX
#ifdef __CUDACC__
#define CUDA_HOST_PREFIX inline __host__
#else
#define CUDA_HOST_PREFIX 
#endif
#endif

namespace multiviewnative {

struct multiview_data {

  std::vector<multiviewnative::image_stack> views_;
  std::vector<multiviewnative::image_stack> kernel1_;
  std::vector<multiviewnative::image_stack> kernel2_;
  std::vector<multiviewnative::image_stack> weights_;
    
  std::vector<int> stack_dims_;
  std::vector<int> kernel1_dims_;
  std::vector<int> kernel2_dims_;

  CUDA_HOST_PREFIX multiview_data(const std::vector<int>& stack_dims, int _n_views = 6):
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
    std::cout << "[multiview_data] " 
	      << nviews << "x views/weights "<< views_[0].shape()[0] << "x"  << views_[0].shape()[1] << "x"  << views_[0].shape()[2] 
	      << " kernel1 "<< kernel1_[0].shape()[0] << "x"  << kernel1_[0].shape()[1] << "x"  << kernel1_[0].shape()[2] 
	      << " kernel2 "<< kernel2_[0].shape()[0] << "x"  << kernel2_[0].shape()[1] << "x"  << kernel2_[0].shape()[2] 
	      << "\n";

  }
};

struct image_kernel_data {
  multiviewnative::image_stack stack_;
  multiviewnative::image_stack kernel_;

  image_kernel_data(const std::vector<unsigned>& _shape):
    stack_(_shape),
    kernel_(){
    
    std::fill(stack_.data(), stack_.data() + stack_.num_elements(), 0);
    for (unsigned i = 0; i < stack_.num_elements(); ++i)
      {
	*(stack_.data() + i) = i;
      }
    
    kernel_.resize(boost::extents[_shape[0]/10][_shape[1]/10][_shape[2]/10]);
    std::fill(kernel_.data(), kernel_.data() + kernel_.num_elements(), 0);
    kernel_[kernel_.shape()[0]/2][kernel_.shape()[1]/2][kernel_.shape()[2]/2] = 1;
  }
    
  void info() {
    std::cout << "[data]\t" 
	      << "stack "  << stack_.shape()[0] << "x"  << stack_.shape()[1] << "x"  << stack_.shape()[2] 
	      << " kernel "<< kernel_.shape()[0] << "x"  << kernel_.shape()[1] << "x"  << kernel_.shape()[2] 
	      << "\n";

  }

};


};
#endif /* _SYNTHETIC_DATA_H_ */

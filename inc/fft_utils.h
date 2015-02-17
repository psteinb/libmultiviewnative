#ifndef _FFT_UTILS_H_
#define _FFT_UTILS_H_
#include <vector>
#include <sstream>
#include <map>
#include "fftw_interface.h"
#include "boost/thread.hpp"
#include "image_stack_utils.h"
#include "plan_store.h"

namespace multiviewnative {
  
template <typename ImageStackT>
class inplace_3d_transform {
  
public:
  
  typedef typename ImageStackT::element image_element_type;
  typedef fftw_api_definitions<image_element_type> fftw_api;
  typedef std::vector<unsigned> shape_type;
  typedef typename fftw_api::real_type real_type;
  typedef typename fftw_api::complex_type complex_type;
  typedef typename fftw_api::plan_type plan_type;

  template <typename ContainerT>
  inplace_3d_transform(const ContainerT& _shape):
    input_shape_(begin(_shape), end(_shape)),
    fftw_shape_(ImageStackT::dimensionality,0)
  {
    adapt_extents_for_fftw_inplace(input_shape_, fftw_shape_);
  }

  /**
     \brief 
     forward r-2-c transform, the method assumes that the incoming ImageStackT has already been padded for the fft
     
     \param[in] _input image buffer to be used as input for the FFT
     \param[in] _plan a custom plan can be supplied

     \return 
     \retval 
     
  */  
  void forward(ImageStackT* _input , 
	       plan_type* _plan = 0){

    
    if(!std::equal(fftw_shape_.cbegin(),fftw_shape_.cend(), _input->shape() )){
      std::ostringstream msg;
      msg << "unable to transform input buffer, received (";
      std::copy(_input->shape(), _input->shape() + ImageStackT::dimensionality,
		std::ostream_iterator<image_element_type>(msg, " "));
      msg << "), expected (";
      std::copy(fftw_shape_.cbegin(),fftw_shape_.cend(),
		std::ostream_iterator<image_element_type>(msg, " "));
      msg << ")\n";
      throw std::runtime_error(msg.str().c_str());
    }

    real_type* fourier_input = (real_type*)_input->data(); 
    complex_type* fourier_output  = (complex_type*)_input->data(); 

    if(!_plan){
      if(!multiviewnative::plan_store<image_element_type>::get()->has_key(input_shape_)){
	multiviewnative::plan_store<image_element_type>::get()->add(input_shape_);
      }
      _plan = multiviewnative::plan_store<image_element_type>::get()->get_forward(input_shape_);
    }
          
    
    fftw_api::reuse_plan_r2c(*_plan, fourier_input, fourier_output);

    
    
  };

  void backward(ImageStackT* _input,
		plan_type* _plan = 0){

  
    complex_type*  fourier_input   =  (complex_type*)_input->data();
    real_type*     fourier_output  =  (real_type*)_input->data();

    if(!_plan){
      if(!multiviewnative::plan_store<image_element_type>::get()->has_key(input_shape_)){
	multiviewnative::plan_store<image_element_type>::get()->add(input_shape_);
      }
      _plan = multiviewnative::plan_store<image_element_type>::get()->get_backward(input_shape_);
    }
          
    
    fftw_api::reuse_plan_c2r(*_plan, fourier_input, fourier_output);

    };
  
  

  ~inplace_3d_transform(){
    
  };

  
  void padd_for_fft(ImageStackT* _input){

    std::vector<unsigned> inplace_shape(ImageStackT::dimensionality);
    shape_type input_shape(_input->shape(), _input->shape() + ImageStackT::dimensionality);


    if(!std::equal(fftw_shape_.cbegin(), fftw_shape_.cend(), _input->shape()))
      adapt_extents_for_fftw_inplace( input_shape, inplace_shape,_input->storage_order());
    else
      inplace_shape = fftw_shape_;

    _input->resize(boost::extents[inplace_shape[0]][inplace_shape[1]][inplace_shape[2]]);

  };

  void resize_after_fft(ImageStackT* _input){

    _input->resize(boost::extents[input_shape_[0]][input_shape_[1]][input_shape_[2]]);

  };

    

  
  bool is_ready_for_fft(ImageStackT* _input){
    
    return std::equal(begin(fftw_shape_), end(fftw_shape_), _input->shape());
  }

  void generate_fftw_ready_shape(ImageStackT* _input, shape_type& _padded_shape){
    
    std::vector<unsigned> padded_shape(ImageStackT::dimensionality,0);
    std::vector<unsigned> input_shape(_input->shape(),_input->shape() + ImageStackT::dimensionality);
    adapt_extents_for_fftw_inplace(input_shape, _padded_shape,_input->storage_order());
  
  }

  
  
  
private:
      
  ImageStackT* input_;
  shape_type input_shape_; 
  shape_type fftw_shape_; 


  void clean_up(){

  };

  

};

template <typename ImageStackT>
class parallel_inplace_3d_transform : public inplace_3d_transform<ImageStackT> {

  static int nthreads_;

public:
  typedef fftw_api_definitions<typename ImageStackT::element> fftw_api;
  typedef typename fftw_api::plan_type plan_type;

  
  template <typename ContainerT>
  parallel_inplace_3d_transform(const ContainerT& _shape):
    inplace_3d_transform<ImageStackT>(_shape)
  {
    
  }

  void forward(ImageStackT* _input , 
	       plan_type* _plan = 0){
    
    inplace_3d_transform<ImageStackT>::forward(_input, _plan);

  };

  void backward(ImageStackT* _input , 
	       plan_type* _plan = 0){
    
    inplace_3d_transform<ImageStackT>::backward(_input,_plan);

  };

  static void set_n_threads(int _nthreads = 0){

    if(_nthreads<0)
      _nthreads = int(boost::thread::hardware_concurrency());

    if(_nthreads > 0){
      parallel_inplace_3d_transform::nthreads_ = _nthreads;
      int success = 0;
      success = fftw_api_definitions<typename ImageStackT::element>::init_threads();
      if(!success){
	std::cerr << "parallel_inplace_3d_transform::set_n_threads\tunable to initialize threads of fftw3\n";
      }
      
      fftw_api_definitions<typename ImageStackT::element>::plan_with_threads(_nthreads);
      
    }
         
  }

  static void clear_threads_data(){

    if(parallel_inplace_3d_transform::_nthreads > 0){
      fftw_api_definitions<typename ImageStackT::element>::cleanup_threads();

    }
  }
};

  template <typename T>
  int parallel_inplace_3d_transform<T>::nthreads_;

}
#endif /* _FFT_UTILS_H_ */

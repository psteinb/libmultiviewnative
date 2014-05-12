#ifndef _FFT_UTILS_H_
#define _FFT_UTILS_H_
#include <vector>
#include "fftw_interface.h"
#include "boost/thread.hpp"
#include "image_stack_utils.h"

namespace multiviewnative {
  


template <typename ImageStackT>
class inplace_3d_transform {
  
public:
  
  typedef fftw_api_definitions<typename ImageStackT::element> fftw_api;
  typedef typename fftw_api::real_type real_type;
  typedef typename fftw_api::complex_type complex_type;

  inplace_3d_transform(ImageStackT* _input):
    input_(_input),
    shape_(ImageStackT::dimensionality,0),
    transform_plan_()
  {
    std::copy(_input->shape(), _input->shape() + ImageStackT::dimensionality, shape_.begin());
    
  }

  

  void forward(){

    padd_for_fft(input_);

    real_type* fourier_input = (real_type*)input_->data(); 
    complex_type* fourier_output  = (complex_type*)input_->data(); 

    transform_plan_ = fftw_api::dft_r2c_3d(shape_[0], shape_[1], shape_[2],
					   fourier_input, fourier_output,
					   FFTW_ESTIMATE);
    
    fftw_api::execute_plan(transform_plan_);

    clean_up();
  };

  void backward(){

    complex_type*  fourier_input   =  (complex_type*)input_->data();
    real_type*     fourier_output  =  (real_type*)input_->data();

    transform_plan_ = fftw_api::dft_c2r_3d(shape_[0], shape_[1], shape_[2],
					   fourier_input, fourier_output,
					   FFTW_ESTIMATE);
    
    fftw_api::execute_plan(transform_plan_);

    clean_up();
  };
  
  

  ~inplace_3d_transform(){
  };

private:
      
  ImageStackT* input_;
  std::vector<typename ImageStackT::size_type> shape_; 
  typename fftw_api::plan_type transform_plan_;
  
  void padd_for_fft(ImageStackT* _input){

    std::vector<unsigned> inplace_extents(ImageStackT::dimensionality);
    adapt_extents_for_fftw_inplace(_input->storage_order(),shape_, inplace_extents);
    _input->resize(boost::extents[inplace_extents[0]][inplace_extents[1]][inplace_extents[2]]);

  };

  void clean_up(){
    fftw_api::destroy_plan(transform_plan_);
  };

};

template <typename ImageStackT>
class parallel_inplace_3d_transform : public inplace_3d_transform<ImageStackT> {

  static int nthreads_;

public:
  parallel_inplace_3d_transform(ImageStackT*  _input):
    inplace_3d_transform<ImageStackT>(_input)
  {
    
  }

  void forward(){
    
    inplace_3d_transform<ImageStackT>::forward();

  };

  void backward(){
    
    inplace_3d_transform<ImageStackT>::backward();

  };

  static void set_n_threads(int _nthreads = 0){

    if(_nthreads<0)
      _nthreads = int(boost::thread::hardware_concurrency());

    if(_nthreads > 0){
      parallel_inplace_3d_transform::nthreads_ = _nthreads;
      int success = fftw_init_threads();
      if(!success){
	std::cerr << "parallel_inplace_3d_transform::set_n_threads\tunable to initialize threads of fftw3\n";
      }
      fftw_plan_with_nthreads(nthreads_);
    }
         
  }

  static void clear_threads_data(){

    if(parallel_inplace_3d_transform::_nthreads > 0){
      fftw_cleanup_threads();
    }
  }
};

  template <typename T>
  int parallel_inplace_3d_transform<T>::nthreads_;

}
#endif /* _FFT_UTILS_H_ */

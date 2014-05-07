#ifndef _CPU_KERNELS_HPP_
#define _CPU_KERNELS_HPP_
#include <cmath>


template <typename TransferT, typename SizeT>
void computeQuotient(const TransferT* _input,TransferT* _output, const SizeT& _size){
  for(SizeT idx = 0;idx<_size;++idx)
    _output[idx] = _input[idx]/_output[idx] ;
}



template <typename TransferT>
struct FinalValues {

  //no ownership is taken
  TransferT*  psi_        ;
  const       TransferT*  integral_  ;
  const       TransferT*  weight_    ;
  size_t       size_       ;
  size_t       offset_     ;
  double      lambda_     ;
  TransferT   minValue_   ;

  typedef void (FinalValues::*member_function)();
  member_function callback_;

  FinalValues(TransferT* _psi,
		     const TransferT* _integral, 
		     const TransferT* _weight, 
		     size_t _size,
		     size_t _offset = 0,
		     double _lambda = 0.006,
		     TransferT _minValue = .0001f):
    psi_       (  _psi       )  ,
    integral_  (  _integral  )  ,
    weight_    (  _weight    )  ,
    size_      (  _size      )  ,
    offset_    (  _offset    )  ,
    lambda_    (  _lambda    )  ,
    minValue_  (  _minValue  )  ,
    callback_   ( &FinalValues::plain )
  {

    if(_lambda>0)
      callback_   = &FinalValues::regularized ;
    
  }


  void compute() {
    (this->*callback_)();
  }

  void plain(){
    TransferT value = 0.f;
    TransferT last_value = 0.f;
    TransferT next_value = 0.f;
    for(size_t pixel = offset_;pixel<size_;++pixel){
      last_value = psi_[pixel];
      value = last_value*integral_[pixel];
      if(!(value>0.f)){
	value = minValue_;
      }
      
      if(std::isnan(value) || std::isinf(value))
	next_value = minValue_;
      else
	next_value = std::max(value,minValue_);

      next_value = weight_[pixel]*(next_value - last_value) + last_value;
      psi_[pixel] = next_value;
    }

  }

  //
  // perform Tikhonov regularization if desired
  //
  void regularized(){
    TransferT value = 0.f;
    TransferT last_value = 0.f;
    TransferT next_value = 0.f;
    TransferT lambda_inv = 1.f / lambda_;

    for(size_t pixel = offset_;pixel<size_;++pixel){
      last_value = psi_[pixel];
      value = last_value*integral_[pixel];
      if(value>0.f){
	value = lambda_inv*(std::sqrt(1. + 2.*lambda_*value ) - 1.)  ;
      }
      else{
	value = minValue_;
      }
      
      if(std::isnan(value) || std::isinf(value))
	next_value = minValue_;
      else
	next_value = std::max(value,minValue_);

      next_value = weight_[pixel]*(next_value - last_value) + last_value;
      psi_[pixel] = next_value;
    }
  }
};

template <typename TransferT>
void computeFinalValues(TransferT* _psi,const TransferT* _integral, const TransferT* _weight, 
			size_t _size,
			size_t _offset,
			double _lambda = 0.006,
			TransferT _minValue = .0001f){
  
  TransferT value = 0.f;
  TransferT last_value = 0.f;
  TransferT next_value = 0.f;
  for(size_t pixel = _offset;pixel<_size;++pixel){
    last_value = _psi[pixel];
    value = last_value*_integral[pixel];
    if(value>0.f){
      //
      // perform Tikhonov regularization if desired
      //
      if(_lambda>0.)
	value = (std::sqrt(1. + 2.*_lambda*value ) - 1.) / _lambda;}
    else{
      value = _minValue;
    }
      
    if(std::isnan(value) || std::isinf(value))
      next_value = _minValue;
    else
      next_value = std::max(value,_minValue);

    next_value = _weight[pixel]*(next_value - last_value) + last_value;
    _psi[pixel] = next_value;
  }

}

#ifdef _OPENMP //macro that is defined if gcc is called with -fopenmp flag
#include <omp.h>
template <typename TransferT>
void computeQuotientMultiCPU(TransferT* _input,TransferT* _output, long long int _size){

  int num_procs = omp_get_num_procs();
  size_t chunk_size = (_size + num_procs -1 )/num_procs;
  size_t idx;

#pragma omp parallel shared(_input,_output,_size,chunk_size) private(idx)
  {

  #pragma omp for schedule(dynamic,chunk_size) nowait
  for(idx = 0;idx<_size;++idx)
    _output[idx] = _input[idx]/_output[idx];

  } 

}

template <typename TransferT>
void computeFinalValuesMultiCPU(TransferT* _image,TransferT* _integral, TransferT* _weight, 
			   size_t _size,
			   size_t _offset,
			   double _lambda,
			   TransferT _minValue){
  
  TransferT value = 0.f;
  TransferT new_value = 0.f;
  size_t pixel = 0;
  int num_procs = omp_get_num_procs();
  size_t chunk_size = ((_size - _offset) + num_procs -1 )/num_procs;
  
#pragma omp parallel shared(_image,_integral,_weight,chunk_size) private(pixel,_offset,_size,_lambda,_minValue,value,new_value)
  {

#pragma omp for schedule(dynamic,chunk_size) nowait
  for(pixel = _offset;pixel<_size;++pixel){
    value = _image[pixel]*_integral[pixel];
    if(value>0.f){
      if(_lambda>0.)
	value = (std::sqrt(1. + 2.*_lambda*value ) - 1.) / _lambda;}
    else{
      value = _minValue;
    }
      
    new_value = std::max(value,_minValue);
    new_value = _weight[pixel]*(new_value - value) + value;
    _image[pixel] = new_value;
  }

  }
}
#endif
#endif /* _COMPUTE_KERNELS_CUH_ */

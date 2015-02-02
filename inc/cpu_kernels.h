#ifndef _CPU_KERNELS_HPP_
#define _CPU_KERNELS_HPP_
#include <cmath>


template <typename TransferT, typename SizeT>
void computeQuotient(const TransferT* _input,TransferT* _output, const SizeT& _size){
  for(SizeT idx = 0;idx<_size;++idx){
    TransferT temp = 1./_output[idx];
    _output[idx] = _input[idx]*temp ;
  }
}

template <typename TransferT>
void serial_final_values(TransferT* _psi,
			 const TransferT* _integral, 
			 const TransferT* _weight, 
			 const size_t& _size,
			 const TransferT& _minValue = 0.0001f,
			 const size_t& _offset = 0){
  using std::isinf;
  using std::isnan;
  
  TransferT value = 0.f;
  TransferT last_value = 0.f;
  TransferT next_value = 0.f;
  for(size_t pixel = _offset;pixel<_size;++pixel){
    last_value = _psi[pixel];
    value = last_value*_integral[pixel];
    if(!(value>0.f)){
      value = _minValue;
    }
      
    if(isnan(value) || isinf(value))
      next_value = _minValue;
    else
      next_value = std::max(value,_minValue);

    next_value = _weight[pixel]*(next_value - last_value) + last_value;
    _psi[pixel] = next_value;
  }

}

//
// perform Tikhonov regularization if desired
//
template <typename TransferT>
void serial_regularized_final_values(TransferT* _psi,
				     const TransferT* _integral, 
				     const TransferT* _weight, 
				     const size_t& _size,
				     const double& _lambda,
				     const TransferT& _minValue = 0.0001f,
				     const size_t& _offset = 0){
  using std::isinf;
  using std::isnan;
  
  TransferT value = 0.f;
  TransferT last_value = 0.f;
  TransferT next_value = 0.f;
  TransferT lambda_inv = 1.f / _lambda;

  for(size_t pixel = _offset;pixel<_size;++pixel){
    last_value = _psi[pixel];
    value = last_value*_integral[pixel];
    if(value>0.f){
      value = lambda_inv*(std::sqrt(1. + 2.*_lambda*value ) - 1.)  ;
    }
    else{
      value = _minValue;
    }
      
    if(isnan(value) || isinf(value))
      next_value = _minValue;
    else
      next_value = std::max(value,_minValue);

    next_value = _weight[pixel]*(next_value - last_value) + last_value;
    _psi[pixel] = next_value;
  }
}

template <typename TransferT>
void computeFinalValues(TransferT* _psi,const TransferT* _integral, const TransferT* _weight, 
			size_t _size,
			size_t _offset,
			double _lambda = 0.006,
			TransferT _minValue = .0001f){
  using std::isinf;
  using std::isnan;

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
      
    if(isnan(value) || isinf(value))
      next_value = _minValue;
    else
      next_value = std::max(value,_minValue);

    next_value = _weight[pixel]*(next_value - last_value) + last_value;
    _psi[pixel] = next_value;
  }

}

#ifdef _OPENMP //macro that is defined if gcc is called with -fopenmp flag
#include <omp.h>
template <typename TransferT, typename SizeT  >
void parallel_divide(const TransferT* _input,TransferT* _output, const SizeT& _size, const int& _num_threads = -1){

  int num_procs = _num_threads > 0 ? _num_threads : omp_get_num_procs();
  SizeT chunk_size = (_size + num_procs -1 )/num_procs;
  SizeT idx;
  omp_set_num_threads(num_procs);

#pragma omp parallel for schedule(dynamic,chunk_size) shared(_output)
  for(idx = 0;idx<_size;++idx){
    TransferT temp = 1./_output[idx];
    _output[idx] = temp*_input[idx];
  }

}

template <typename TransferT>
void parallel_final_values(TransferT* _psi,
			   const TransferT* _integral, 
			   const TransferT* _weight, 
			   const size_t& _size,
			   const int& _num_threads = -1,
			   const TransferT& _minValue = 0.0001f,
			   const size_t& _offset = 0){

  using std::isinf;
  using std::isnan;

  int num_procs = (_num_threads > 0) ? _num_threads : omp_get_num_procs();
  omp_set_num_threads(num_procs);

  size_t chunk_size = ((_size - _offset) + num_procs - 1 )/num_procs;
  size_t pixel = 0;

#pragma omp parallel for schedule(dynamic,chunk_size) shared(_psi) private(pixel)
  for(pixel = _offset;pixel<_size;++pixel){
    TransferT last_value = _psi[pixel];
    TransferT value = last_value*_integral[pixel];
    if(!(value>0.f)){
      value = _minValue;
    }
      
    TransferT next_value = 0;
    if(isnan(value) || isinf(value))
      next_value = _minValue;
    else
      next_value = std::max(value,_minValue);

    next_value = _weight[pixel]*(next_value - last_value) + last_value;
    _psi[pixel] = next_value;
  }

}

//
// perform Tikhonov regularization if desired
//
template <typename TransferT>
void parallel_regularized_final_values(TransferT* _psi,
				       const TransferT* _integral, 
				       const TransferT* _weight, 
				       const size_t& _size,
				       double _lambda = 0.006,
				       const int& _num_threads = -1,
				       const TransferT& _minValue = 0.0001f,
				       const size_t& _offset = 0){
  using std::isinf;
  using std::isnan;
  
  TransferT lambda_inv = 1.f / _lambda;
  int num_procs = (_num_threads > 0) ? _num_threads : omp_get_num_procs();
  size_t chunk_size = ((_size - _offset) + num_procs - 1 )/num_procs;
  omp_set_num_threads(num_procs);
    

  size_t pixel = 0;
#pragma omp parallel for schedule(dynamic,chunk_size) shared(_psi) private(pixel)
  for(pixel = _offset  ;pixel<_size;++pixel){
    TransferT last_value = _psi[pixel];
    TransferT value = last_value*_integral[pixel];
    if(value>0.f){
      value = lambda_inv*(std::sqrt(1. + 2.*_lambda*value ) - 1.)  ;
    }
    else{
      value = _minValue;
    }
      
    TransferT next_value = 0;
    if(isnan(value) || isinf(value))
      next_value = _minValue;
    else
      next_value = std::max(value,_minValue);

    next_value = _weight[pixel]*(next_value - last_value) + last_value;
    _psi[pixel] = next_value;
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
  
#pragma omp parallel for schedule(dynamic,chunk_size) shared(_image) private(pixel)
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
#endif
#endif /* _CPU_KERNELS_H_ */

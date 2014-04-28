#ifndef _CPU_KERNELS_HPP_
#define _CPU_KERNELS_HPP_
#include <cmath>


template <typename TransferT, typename SizeT>
void computeQuotient(const TransferT* _input,TransferT* _output, const SizeT& _size){
  for(SizeT idx = 0;idx<_size;++idx)
    _output[idx] = ( _output[idx]!= 0.f ) ? _input[idx]/_output[idx] : _output[idx];
}






template <typename TransferT>
void computeFinalValues(TransferT* _psi,const TransferT* _integral, const TransferT* _weight, 
			size_t _size,
			size_t _offset,
			double _lambda = 0.006,
			TransferT _minValue = .0001f){
  
  TransferT value = 0.f;
  TransferT last_value = 0.f;
  TransferT new_value = 0.f;
  for(size_t pixel = _offset;pixel<_size;++pixel){
    last_value = _psi[pixel];
    value = _psi[pixel]*_integral[pixel];
    if(value>0.f){
      //
      // perform Tikhonov regularization if desired
      //
      if(_lambda>0.)
	value = (std::sqrt(1. + 2.*_lambda*value ) - 1.) / _lambda;}
    else{
      value = _minValue;
    }
      
    if(std::isnan(value))
      new_value = _minValue;
    else
      new_value = std::max(value,_minValue);

    new_value = _weight[pixel]*(new_value - last_value) + value;
    _psi[pixel] = new_value;
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

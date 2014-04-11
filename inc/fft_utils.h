#ifndef _FFT_UTILS_H_
#define _FFT_UTILS_H_
#include <vector>
#include "fftw_interface.h"

namespace multiviewnative {
  template <typename StorageT, typename DimT, typename ODimT>
  void adapt_extents_for_fftw_inplace(const StorageT& _storage, const DimT& _extent, ODimT& _value){

    std::fill(_value.begin(),_value.end(),0);

    std::vector<int> storage_order(_extent.size());
    for(size_t i = 0;i<_extent.size();++i)
      storage_order[i] = _storage.ordering(i);

    int lowest_storage_index = std::min_element(storage_order.begin(),storage_order.end()) - storage_order.begin() ;

    for(size_t i = 0;i<_extent.size();++i)
      _value[i] = (lowest_storage_index == i) ? 2*(_extent[i]/2 + 1) : _extent[i];  
  
  
  }


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


}
#endif /* _FFT_UTILS_H_ */

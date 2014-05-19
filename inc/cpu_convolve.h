#ifndef _CPU_CONVOLVE_H_
#define _CPU_CONVOLVE_H_

#include <algorithm>
#include <numeric>
#include <iostream> 
#include <iomanip> 
#include <vector>

#include "padd_utils.h"
#include "boost/multi_array.hpp"
#include "fft_utils.h"
#include "image_stack_utils.h"

namespace multiviewnative {

  typedef multiviewnative::zero_padd<multiviewnative::image_stack> wrap_around_padding;
    

  template <template<class> class TransformT = multiviewnative::inplace_3d_transform, 
	    typename PaddingT = wrap_around_padding, 
	    typename TransferT = float, 
	    typename SizeT = unsigned>
  struct cpu_convolve : public PaddingT {

    typedef TransferT value_type;
    typedef SizeT size_type;
    typedef PaddingT padding_policy;
    typedef boost::multi_array<value_type,3, fftw_allocator<value_type> >    fftw_image_stack;
    typedef TransformT<fftw_image_stack> transform_policy;
    
    static const int num_dims = 3;

    image_stack_ref* image_;
    fftw_image_stack* padded_image_;

    image_stack_ref* kernel_;
    fftw_image_stack* padded_kernel_;
    
    cpu_convolve(value_type* _image_stack_arr, size_type* _image_extents_arr,
		 value_type* _kernel_stack_arr, size_type* _kernel_extents_arr, 
		 size_type* _storage_order = 0):
      PaddingT(&_image_extents_arr[0],&_kernel_extents_arr[0]),
      image_(0),
      padded_image_(0),
      kernel_(0),
      padded_kernel_(0)
    {
      std::vector<size_type> image_shape(num_dims);
      std::copy(_image_extents_arr, _image_extents_arr+num_dims,image_shape.begin());

      std::vector<size_type> kernel_shape(num_dims);
      std::copy(_kernel_extents_arr, _kernel_extents_arr+num_dims,kernel_shape.begin());

      
      multiviewnative::storage local_order = boost::c_storage_order();
      if(_storage_order){
	bool ascending[3] = {true, true, true};
	local_order = storage(_storage_order,ascending);
      }
      
      this->image_ = new multiviewnative::image_stack_ref(_image_stack_arr, image_shape, local_order);
      this->kernel_ = new multiviewnative::image_stack_ref(_kernel_stack_arr, kernel_shape, local_order);
      
      this->padded_image_ = new fftw_image_stack(this->extents_, local_order);
      this->padded_kernel_ = new fftw_image_stack(this->extents_, local_order);
      
      this->insert_at_offsets(*image_,*padded_image_);
      this->wrapped_insert_at_offsets(*kernel_,*padded_kernel_);

    };

    void inplace(){
      
      typedef typename TransformT<fftw_image_stack>::complex_type complex_type;

      TransformT<fftw_image_stack> image_transform(padded_image_);
      TransformT<fftw_image_stack> kernel_transform(padded_kernel_);
      
      image_transform.forward();
      kernel_transform.forward();

      complex_type*  complex_image_fourier   =  (complex_type*)padded_image_->data();
      complex_type*  complex_kernel_fourier  =  (complex_type*)padded_kernel_->data();

      unsigned fourier_num_elements = padded_image_->num_elements()/2;
      for(unsigned long index = 0;index < fourier_num_elements;++index){
	value_type real = complex_image_fourier[index][0]*complex_kernel_fourier[index][0] - complex_image_fourier[index][1]*complex_kernel_fourier[index][1];
	value_type imag = complex_image_fourier[index][0]*complex_kernel_fourier[index][1] + complex_image_fourier[index][1]*complex_kernel_fourier[index][0];
	complex_image_fourier[index][0] = real;
	complex_image_fourier[index][1] = imag;
      }

      image_transform.backward();

      size_type transform_size = std::accumulate(this->extents_.begin(),this->extents_.end(),1,std::multiplies<size_type>());
      value_type scale = 1.0 / (transform_size);
      for(unsigned long index = 0;index < padded_image_->num_elements();++index){
	padded_image_->data()[index]*=scale;
      }

      (*image_) = (*padded_image_)[ boost::indices[range(this->offsets_[0], this->offsets_[0]+image_->shape()[0])][range(this->offsets_[1], this->offsets_[1]+image_->shape()[1])][range(this->offsets_[2], this->offsets_[2]+image_->shape()[2])] ];
    };

    ~cpu_convolve(){
      delete image_;
      delete kernel_;
      delete padded_image_;
      delete padded_kernel_;
    };

  private:
        

  };

}

#endif /* _CPU_CONVOLVE_H_ */




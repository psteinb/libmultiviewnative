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
  typedef multiviewnative::no_padd<multiviewnative::image_stack> no_padding;
  typedef boost::multi_array<float,3, fftw_allocator<float> >    fftw_image_stack;

  template <template<class> class TransformT = multiviewnative::inplace_3d_transform, 
	    typename PaddingT = no_padding,//wrap_around_padding, 
	    typename TransferT = float, 
	    typename SizeT = unsigned>
  struct cpu_convolve : public PaddingT {

    typedef TransferT value_type;
    typedef SizeT size_type;
    typedef PaddingT padding_policy;
    typedef TransformT<fftw_image_stack> transform_policy;
    
    static const int num_dims = image_stack_ref::dimensionality;

    image_stack_ref* image_;
    fftw_image_stack* padded_image_;

    image_stack_ref* kernel_;
    fftw_image_stack* padded_kernel_;

    transform_policy fft;
    
    /**
       \brief the content of _image_stack_arr and _kernel_stack_arr is expected in row major memory order, however the extents are expected to contain the shape of the stack conforming this convention, so {z-dim, y-dim, x-dim}. 
       For example, in case of a stack of 
       width = 2
       height = 3
       depth = 4
       the shape buffer is expected to contain
       size_type* _image_extents_arr = {4,3,2};
       
       \param[in] _image_stack_arr buffer containing an 3D image stack of shape _image_extents_arr
       \param[in] _kernel_extents_arr containing the shape of the kernel to receive

       \return 
       \retval 
       
    */
    template <typename int_type>
    cpu_convolve(value_type* _image_stack_arr, int_type* _image_extents_arr,
		 int_type* _kernel_extents_arr = 0, 
		 int_type* _storage_order = 0):
      PaddingT(&_image_extents_arr[0],&_kernel_extents_arr[0]),
      image_(0),
      padded_image_(0),
      kernel_(0),
      padded_kernel_(0),
      fft()
    {
      std::vector<size_type> image_shape(_image_extents_arr, _image_extents_arr+num_dims);
      std::vector<size_type> kernel_shape(_kernel_extents_arr, _kernel_extents_arr+num_dims);
      multiviewnative::storage local_order = boost::c_storage_order();
      if(_storage_order){
	bool ascending[3] = {true, true, true};
	local_order = storage(_storage_order,ascending);
      }

      this->image_ = new multiviewnative::image_stack_ref(_image_stack_arr, image_shape, local_order);
      this->padded_image_ = new fftw_image_stack(this->extents_, local_order);
      
      this->insert_at_offsets(*image_,*padded_image_);

      fft = transform_policy(this->extents_);
    }
    /**
       \brief the content of _image_stack_arr and _kernel_stack_arr is expected in row major memory order, however the extents are expected to contain the shape of the stack conforming this convention, so {z-dim, y-dim, x-dim}. 
       For example, in case of a stack of 
       width = 2
       height = 3
       depth = 4
       the shape buffer is expected to contain
       size_type* _image_extents_arr = {4,3,2};
       
       \param[in] _image_stack_arr buffer containing an 3D image stack of shape _image_extents_arr
       \param[in] _kernel_stack_arr buffer containing an 3D kernel stack of shape _kernel_extents_arr
       
       \return 
       \retval 
       
    */
    template <typename int_type>
    cpu_convolve(value_type* _image_stack_arr, int_type* _image_extents_arr,
		 value_type* _kernel_stack_arr = 0, int_type* _kernel_extents_arr = 0, 
		 size_type* _storage_order = 0):
      PaddingT(&_image_extents_arr[0],&_kernel_extents_arr[0]),
      image_(0),
      padded_image_(0),
      kernel_(0),
      padded_kernel_(0),
      fft()
    {
      
      std::vector<size_type> image_shape(_image_extents_arr, _image_extents_arr+num_dims);
      std::vector<size_type> kernel_shape(_kernel_extents_arr, _kernel_extents_arr+num_dims);
      
      multiviewnative::storage local_order = boost::c_storage_order();
      if(_storage_order){
	bool ascending[3] = {true, true, true};
	local_order = storage(_storage_order,ascending);
      }
      
      this->image_ = new multiviewnative::image_stack_ref(_image_stack_arr, image_shape, local_order);
      this->padded_image_ = new fftw_image_stack(this->extents_, local_order);

      
      this->kernel_ = new multiviewnative::image_stack_ref(_kernel_stack_arr, kernel_shape, local_order);
      this->padded_kernel_ = new fftw_image_stack(this->extents_, local_order);

      
      this->insert_at_offsets(*image_,*padded_image_);
      this->wrapped_insert_at_offsets(*kernel_,*padded_kernel_);
      
      fft = transform_policy(this->extents_);
    };


    void inplace(){
      
      typedef typename TransformT<fftw_image_stack>::complex_type complex_type;

      // shape_t tx_shape(padded_image_->shape(), padded_image_->shape() + num_dims);

      // //set up transforms
      // TransformT<fftw_image_stack> stack_transform(tx_shape);
          
      fft.padd_for_fft(padded_image_ );
      fft.forward( padded_image_  );

      fft.padd_for_fft(padded_kernel_ );
      fft.forward( padded_kernel_  );      

      complex_type*  complex_image_fourier   =  (complex_type*)padded_image_->data();
      complex_type*  complex_kernel_fourier  =  (complex_type*)padded_kernel_->data();

      unsigned fourier_num_elements = padded_image_->num_elements()/2;
      for(unsigned long index = 0;index < fourier_num_elements;++index){
	value_type real = complex_image_fourier[index][0]*complex_kernel_fourier[index][0] - complex_image_fourier[index][1]*complex_kernel_fourier[index][1];
	value_type imag = complex_image_fourier[index][0]*complex_kernel_fourier[index][1] + complex_image_fourier[index][1]*complex_kernel_fourier[index][0];
	complex_image_fourier[index][0] = real;
	complex_image_fourier[index][1] = imag;
      }
      
      fft.backward(padded_image_);
      fft.resize_after_fft(padded_image_);

      size_type transform_size = std::accumulate(this->extents_.begin(),this->extents_.end(),1,std::multiplies<size_type>());
      value_type scale = 1.0 / (transform_size);
      for(unsigned long index = 0;index < padded_image_->num_elements();++index){
	padded_image_->data()[index]*=scale;
      }

      if(!std::equal(image_->shape(), 
		     image_->shape() + num_dims,
		     padded_image_->shape()
		     )
	 )
	(*image_) = (*padded_image_)[ boost::indices[range(this->offsets_[0], this->offsets_[0]+image_->shape()[0])][range(this->offsets_[1], this->offsets_[1]+image_->shape()[1])][range(this->offsets_[2], this->offsets_[2]+image_->shape()[2])] ];
      else
	(*image_) = (*padded_image_);

    };


    /**
       \brief contract is equal to the one of inplace, except that this function expects the padded_kernel buffer is handed in through the function parameters (it is expected to have the same dimensions of the padded image, if not an exception is thrown)
       
       \param[in] _forwarded_padded_kernel forwarded kernel in the same data structure that is used internally
       
       \return 
       \retval 
       
    */
    void half_inplace(const fftw_image_stack& _forwarded_padded_kernel){
      
      typedef typename TransformT<fftw_image_stack>::complex_type complex_type;
      typedef decltype(*_forwarded_padded_kernel.shape()) shape_iter_type;
          
      fft.padd_for_fft(padded_image_ );
      fft.forward( padded_image_  );

      if(!std::equal(_forwarded_padded_kernel.shape(), _forwarded_padded_kernel.shape() + fftw_image_stack::dimensionality,
		     padded_image_->shape())){


	std::ostringstream what_msg;
	auto printer = [&](const shape_iter_type& item){ what_msg << item << " ";};
	
	what_msg << "cpu_convolve::half_inplace\t"
		 << "kernel buffer received is ill shaped for convolution "
		 << "received: " ;
	
	std::for_each(_forwarded_padded_kernel.shape(), _forwarded_padded_kernel.shape() + fftw_image_stack::dimensionality,
		      printer
		      );
	what_msg << ", expected: ";
	std::for_each(padded_image_->shape(), padded_image_->shape() + fftw_image_stack::dimensionality,
		      printer
		      );
	
	std::length_error err(what_msg.str());
	throw err;
      }

      complex_type*  complex_image_fourier   =  (complex_type*)padded_image_->data();
      complex_type*  complex_kernel_fourier  =  (complex_type*)_forwarded_padded_kernel.data();

      unsigned fourier_num_elements = padded_image_->num_elements()/2;
      for(unsigned long index = 0;index < fourier_num_elements;++index){
	value_type real = complex_image_fourier[index][0]*complex_kernel_fourier[index][0] - complex_image_fourier[index][1]*complex_kernel_fourier[index][1];
	value_type imag = complex_image_fourier[index][0]*complex_kernel_fourier[index][1] + complex_image_fourier[index][1]*complex_kernel_fourier[index][0];
	complex_image_fourier[index][0] = real;
	complex_image_fourier[index][1] = imag;
      }
      
      fft.backward(padded_image_);
      fft.resize_after_fft(padded_image_);

      size_type transform_size = std::accumulate(this->extents_.begin(),this->extents_.end(),1,std::multiplies<size_type>());
      value_type scale = 1.0 / (transform_size);
      for(unsigned long index = 0;index < padded_image_->num_elements();++index){
	padded_image_->data()[index]*=scale;
      }

      if(!std::equal(image_->shape(), 
		     image_->shape() + num_dims,
		     padded_image_->shape()
		     )
	 )
	(*image_) = (*padded_image_)[ boost::indices[range(this->offsets_[0], this->offsets_[0]+image_->shape()[0])][range(this->offsets_[1], this->offsets_[1]+image_->shape()[1])][range(this->offsets_[2], this->offsets_[2]+image_->shape()[2])] ];
      else
	(*image_) = (*padded_image_);

    };    

    ~cpu_convolve(){
      if(image_)
	delete image_;
      
      if(kernel_)
	delete kernel_;

      if(padded_image_)
	 delete padded_image_;
      
      if(padded_kernel_)
	delete padded_kernel_;
    };

  private:
        

  };

}

#endif /* _CPU_CONVOLVE_H_ */




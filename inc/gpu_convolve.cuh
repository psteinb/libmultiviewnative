#ifndef _GPU_CONVOLVE_CUH_
#define _GPU_CONVOLVE_CUH_

#include <algorithm>
#include <numeric>
#include <iostream> 
#include <iomanip> 
#include <vector>


#include "boost/multi_array.hpp"
#include "cufft_utils.cuh"
#include "image_stack_utils.h"
#include "cuda_kernels.cuh"
#include "cuda_helpers.cuh"

namespace multiviewnative {

  typedef  boost::multi_array<float,              3>    image_stack;
  typedef  boost::multi_array_ref<float,          3>    image_stack_ref;
  typedef  image_stack::array_view<3>::type		image_stack_view;
  typedef  boost::multi_array_types::index_range	range;
  typedef  boost::general_storage_order<3>		storage;

  template <typename PaddingT, typename TransferT, typename SizeT>
  struct gpu_convolve : public PaddingT {

    typedef TransferT value_type;
    typedef SizeT size_type;

    
    
    gpu_convolve(value_type* _image_stack_arr, size_type* _image_extents_arr,
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
      
      //TODO: the following could be on on_device as well
      this->padded_image_ = new multiviewnative::image_stack(this->extents_, local_order);
      this->padded_kernel_ = new multiviewnative::image_stack(this->extents_, local_order);
      
      this->insert_at_offsets(*image_,*padded_image_);
      this->wrapped_insert_at_offsets(*kernel_,*padded_kernel_);

    };

    template <typename TransformT>
    void inplace_on_device(value_type* _image_on_device, value_type* _kernel_on_device){
            

      TransformT image_transform(_image_on_device, &(this->extents_[0]));
      TransformT kernel_transform(_kernel_on_device, &(this->extents_[0]));
      image_transform.forward();
      kernel_transform.forward();

      size_type transform_size = std::accumulate(this->extents_.begin(),this->extents_.end(),1,std::multiplies<size_type>());      
      unsigned fourier_num_elements = padded_image_->num_elements()/2;
      value_type scale = 1.0 / (transform_size);

      unsigned numThreads = 32;
      unsigned numBlocks = largestDivisor(fourier_num_elements,numThreads);

      modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)_image_on_device, (cufftComplex *)_kernel_on_device, fourier_num_elements, scale);
      HANDLE_ERROR(cudaPeekAtLastError());

      
      image_transform.backward();

      
    };

     template <typename TransformT>
     void inplace(){

       //extend kernel and image according to inplace requirements (docs.nvidia.com/cuda/cufft/index.html#multi-dimensional)
       std::vector<unsigned> inplace_extents(this->extents_.size());
       adapt_extents_for_fftw_inplace(padded_image_->storage_order(),this->extents_, inplace_extents);
       padded_image_->resize(boost::extents[inplace_extents[0]][inplace_extents[1]][inplace_extents[2]]);
       padded_kernel_->resize(boost::extents[inplace_extents[0]][inplace_extents[1]][inplace_extents[2]]);

       //place image and kernel on device
       size_type fft_size_byte = padded_image_->num_elements()*sizeof(value_type);
       value_type* image_on_device = 0;
       value_type* kernel_on_device = 0;

       HANDLE_ERROR( cudaMalloc( (void**)&(image_on_device), fft_size_byte ) );
       HANDLE_ERROR( cudaMalloc( (void**)&(kernel_on_device), fft_size_byte ) );

       HANDLE_ERROR( cudaMemcpy( image_on_device, padded_image_->data(), fft_size_byte , cudaMemcpyHostToDevice ) );       
       HANDLE_ERROR( cudaMemcpy( kernel_on_device, padded_kernel_->data(), fft_size_byte , cudaMemcpyHostToDevice ) );

       //perform transform
       this->inplace_on_device<TransformT>(image_on_device,kernel_on_device);

       //place image and kernel on device
       HANDLE_ERROR( cudaMemcpy(padded_image_->data(), image_on_device , fft_size_byte , cudaMemcpyDeviceToHost ) );

       //cut-out region of interest
       (*image_) = (*padded_image_)[ boost::indices[range(this->offsets_[0], this->offsets_[0]+image_->shape()[0])][range(this->offsets_[1], this->offsets_[1]+image_->shape()[1])][range(this->offsets_[2], this->offsets_[2]+image_->shape()[2])] ];
     };

    ~gpu_convolve(){
      delete image_;
      delete kernel_;
      delete padded_image_;
      delete padded_kernel_;
    };

    void set_device(const int& _device){
      HANDLE_ERROR( cudaSetDevice( _device ) );
    };
    
  private:
        
    static const int num_dims = 3;

    image_stack_ref* image_;
    image_stack* padded_image_;

    image_stack_ref* kernel_;
    image_stack* padded_kernel_;

  };

}

#endif /* _GPU_CONVOLVE_H_ */




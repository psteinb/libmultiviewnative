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
      padded_kernel_(0),
      streams_(2)
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

    template <typename TransformT, typename ExtentsT>
    void inplace_on_device(value_type* _image_on_device, value_type* _kernel_on_device, const ExtentsT& _fft_extents){
            

      TransformT image_transform(_image_on_device, &(this->extents_[0]));
      TransformT kernel_transform(_kernel_on_device, &(this->extents_[0]));
      image_transform.forward(&streams_[0]);
      kernel_transform.forward(&streams_[1]);

      size_type transform_size = std::accumulate(this->extents_.begin(),this->extents_.end(),1,std::multiplies<size_type>());      
      unsigned fft_num_elements = std::accumulate(_fft_extents.begin(),_fft_extents.end(),1,std::multiplies<size_type>())/2;

      value_type scale = 1.0 / (transform_size);

      unsigned numThreads = 32;
      unsigned numBlocks = largestDivisor(fft_num_elements,numThreads);

      cudaStreamSynchronize(streams_[0]);
      modulateAndNormalize_kernel<<<numBlocks,numThreads,0,streams_[1]>>>((cufftComplex *)_image_on_device, 
									   (cufftComplex *)_kernel_on_device, 
									   fft_num_elements, 
									   scale);
      HANDLE_ERROR(cudaPeekAtLastError());
      
      image_transform.backward(&streams_[1]);
      
    }

     template <typename TransformT>
     void inplace(){

       //extend kernel and image according to inplace requirements (docs.nvidia.com/cuda/cufft/index.html#multi-dimensional)
       std::vector<unsigned> inplace_extents(this->extents_.size());
       adapt_extents_for_fftw_inplace(padded_image_->storage_order(),this->extents_, inplace_extents);
       //TODO: depending on the Transform, the inplace extents need to be propagated to the transform 
       //(inplace transforms by CUFFT in native mode do not follow the same data conventions than fftw)
       
       //place image and kernel on device
       size_type padded_size_byte = std::accumulate(inplace_extents.begin(), inplace_extents.end(),1,std::multiplies<unsigned>())*sizeof(value_type);
       value_type* image_on_device = 0;
       value_type* kernel_on_device = 0;

       for (int i = 0; i < 2; ++i)
	 HANDLE_ERROR(cudaStreamCreate(&streams_[i]));

       HANDLE_ERROR( cudaMalloc( (void**)&(image_on_device), padded_size_byte ) );
       HANDLE_ERROR( cudaMalloc( (void**)&(kernel_on_device), padded_size_byte ) );

       HANDLE_ERROR( cudaMemcpyAsync( image_on_device, padded_image_->data(), 
				      padded_image_->num_elements()*sizeof(value_type) , 
				      cudaMemcpyHostToDevice, streams_[0] ) );       

       HANDLE_ERROR( cudaMemcpyAsync( kernel_on_device, padded_kernel_->data(), 
				      padded_kernel_->num_elements()*sizeof(value_type) , 
				      cudaMemcpyHostToDevice, streams_[1] ) );
       
       this->inplace_on_device<TransformT>(image_on_device,kernel_on_device,inplace_extents);
       

       HANDLE_ERROR( cudaMemcpyAsync(padded_image_->data(), image_on_device , 
				     padded_image_->num_elements()*sizeof(value_type) , 
				     cudaMemcpyDeviceToHost, streams_[0] ) );

       //cut-out region of interest
       (*image_) = (*padded_image_)[ boost::indices[range(this->offsets_[0], this->offsets_[0]+image_->shape()[0])][range(this->offsets_[1], this->offsets_[1]+image_->shape()[1])][range(this->offsets_[2], this->offsets_[2]+image_->shape()[2])] ];
     
       HANDLE_ERROR( cudaFree( image_on_device ) );
       HANDLE_ERROR( cudaFree( kernel_on_device ) );
 
     }

    ~gpu_convolve(){
      delete image_;
      delete kernel_;
      delete padded_image_;
      delete padded_kernel_;

      for (int i = 0; i < 2; ++i)
	HANDLE_ERROR(cudaStreamDestroy(streams_[i]));

    }

    void set_device(const int& _device){
      HANDLE_ERROR( cudaSetDevice( _device ) );
    };
    
  private:
        
    static const int num_dims = 3;

    image_stack_ref* image_;
    image_stack* padded_image_;

    image_stack_ref* kernel_;
    image_stack* padded_kernel_;

    std::vector<cudaStream_t> streams_;

  };

}

#endif /* _GPU_CONVOLVE_H_ */




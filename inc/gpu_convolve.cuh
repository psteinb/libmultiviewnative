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
#include "cuda_memory.cuh"

namespace multiviewnative {

typedef multiviewnative::stack_on_device<multiviewnative::image_stack,
                                         multiviewnative::asynch>
    asynch_stack_on_device;
typedef multiviewnative::stack_on_device<
    multiviewnative::image_stack, multiviewnative::synch> synch_stack_on_device;

template <typename TransformT, typename TransferT, typename DimT>
void inplace_asynch_convolve_on_device(
    TransferT* _image_on_device, TransferT* _kernel_on_device, DimT* _extents,
    const unsigned& _fft_num_elements,
    const std::vector<cudaStream_t*>& _streams) {

  TransformT image_transform(_image_on_device, _extents);
  TransformT kernel_transform(_kernel_on_device, _extents);
  image_transform.forward(_streams[0]);
  kernel_transform.forward(_streams[1]);

  DimT transform_size =
      std::accumulate(_extents, _extents + 3, 1, std::multiplies<DimT>());
  unsigned eff_fft_num_elements = _fft_num_elements / 2;

  TransferT scale = 1.0 / (transform_size);

  unsigned numThreads = 128;
  unsigned numBlocks = largestDivisor(eff_fft_num_elements, numThreads);

  HANDLE_ERROR(cudaStreamSynchronize(*_streams[1]));
  HANDLE_ERROR(cudaStreamSynchronize(*_streams[0]));
  modulateAndNormalize_kernel << <numBlocks, numThreads>>>
      ((cufftComplex*)_image_on_device, (cufftComplex*)_kernel_on_device,
       eff_fft_num_elements, scale);
  HANDLE_ERROR(cudaPeekAtLastError());

  image_transform.backward(_streams[0]);
}

template <typename TransformT, typename TransferT, typename DimT>
void inplace_convolve_on_device(TransferT* _image_on_device,
                                TransferT* _kernel_on_device, DimT* _extents,
                                const unsigned& _fft_num_elements) {

  TransformT image_transform(_image_on_device, _extents);
  TransformT kernel_transform(_kernel_on_device, _extents);
  image_transform.forward();
  kernel_transform.forward();

  DimT transform_size =
      std::accumulate(_extents, _extents + 3, 1, std::multiplies<DimT>());
  unsigned eff_fft_num_elements = _fft_num_elements / 2;

  TransferT scale = 1.0 / (transform_size);

  unsigned numThreads = 128;
  unsigned numBlocks = largestDivisor(eff_fft_num_elements, numThreads);

  modulateAndNormalize_kernel << <numBlocks, numThreads>>>
      ((cufftComplex*)_image_on_device, (cufftComplex*)_kernel_on_device,
       eff_fft_num_elements, scale);
  HANDLE_ERROR(cudaPeekAtLastError());

  image_transform.backward();
}

template <typename PaddingT, typename TransferT, typename SizeT>
struct gpu_convolve : public PaddingT {

  typedef TransferT value_type;
  typedef SizeT size_type;
  typedef PaddingT padding_policy;
  typedef multiviewnative::image_stack stack_t;
  typedef multiviewnative::image_stack_ref stack_ref_t;

  static const int num_dims = stack_ref_t::dimensionality;


  gpu_convolve()
      : PaddingT(),
        image_(0),
        padded_image_(0),
        kernel_(0),
        padded_kernel_(0),
	cufft_shape_(num_dims,0)// ,
        // streams_(2) 
  {
  }


  //////////////////////////////////////////////////////////////////////////
  // CONSTRUCTORS OPERATE ON HOST DATA
  
  gpu_convolve(value_type* _image_stack_arr, size_type* _image_extents_arr,
               value_type* _kernel_stack_arr, size_type* _kernel_extents_arr,
               size_type* _storage_order = 0)
      : PaddingT(&_image_extents_arr[0], &_kernel_extents_arr[0]),
        image_(0),
        padded_image_(0),
        kernel_(0),
        padded_kernel_(0),
	cufft_shape_(num_dims,0)// ,
        // streams_(2) 
  {
    std::vector<size_type> image_shape(num_dims);
    std::copy(_image_extents_arr, _image_extents_arr + num_dims,
              image_shape.begin());

    std::vector<size_type> kernel_shape(num_dims);
    std::copy(_kernel_extents_arr, _kernel_extents_arr + num_dims,
              kernel_shape.begin());

    multiviewnative::storage local_order = boost::c_storage_order();
    if (_storage_order) {
      bool ascending[3] = {true, true, true};
      local_order = storage(_storage_order, ascending);
    }

    this->image_ = new stack_ref_t(
        _image_stack_arr, image_shape, local_order);
    this->kernel_ = new stack_ref_t(
        _kernel_stack_arr, kernel_shape, local_order);

    
    adapt_extents_for_cufft_inplace(this->extents_, cufft_shape_);
    
    this->padded_image_ =
        new stack_t(this->extents_, local_order);
    this->padded_kernel_ =
        new stack_t(this->extents_, local_order);

    this->insert_at_offsets(*image_, *padded_image_);
    this->wrapped_insert_at_offsets(*kernel_, *padded_kernel_);

    
  };


    template <typename int_type>
    gpu_convolve(value_type* _image_stack_arr,		//
		 int_type* _image_extents_arr,		//
		 int_type* _kernel_extents_arr, 	//
		 size_type* _storage_order = 0)		//
      : PaddingT(&_image_extents_arr[0], &_kernel_extents_arr[0]),
        image_(0),
        padded_image_(0),
        kernel_(0),
        padded_kernel_(0),
	cufft_shape_(num_dims,0)
  {

    std::vector<size_type> image_shape(_image_extents_arr,
                                       _image_extents_arr + num_dims);
    std::vector<size_type> kernel_shape(_kernel_extents_arr,
                                        _kernel_extents_arr + num_dims);

    multiviewnative::storage local_order = boost::c_storage_order();
    if (_storage_order) {
      bool ascending[3] = {true, true, true};
      local_order = storage(_storage_order, ascending);
    }

    this->image_ = new stack_ref_t(
        _image_stack_arr, image_shape, local_order);



    this->padded_image_ = new stack_t(this->extents_, local_order);

    this->insert_at_offsets(*image_, *padded_image_);

    adapt_extents_for_cufft_inplace(this->extents_, cufft_shape_);
    this->padded_image_->resize(cufft_shape_);
    
  }

  template <typename TransformT>
  void inplace() {

    // extend kernel and image according to inplace requirements
    // (docs.nvidia.com/cuda/cufft/index.html#multi-dimensional)
    this->padded_image_->resize(cufft_shape_);
    this->padded_kernel_->resize(cufft_shape_);
    size_type device_memory_elements_required =
        std::accumulate(cufft_shape_.begin(), cufft_shape_.end(), 1,
                        std::multiplies<size_type>());

    // place image and kernel on device (add extra space on device for cufft
    // processing)
    // asynch_stack_on_device
    // image_on_device(padded_image_,device_memory_elements_required);
    // asynch_stack_on_device
    // kernel_on_device(padded_kernel_,device_memory_elements_required);

    unsigned long padded_size_byte =
        padded_image_->num_elements() * sizeof(value_type);

    multiviewnative::device_memory_ports<value_type, 2> device_ports;
    device_ports.create_all_ports(padded_size_byte);
    static const int image_id = 0;
    static const int kernel_id = 1;

    device_ports.template add_stream_for<image_id>();
    device_ports.add_stream_for(kernel_id);

    HANDLE_ERROR(cudaHostRegister(padded_image_->data(), padded_size_byte,
                                  cudaHostRegisterPortable));
    HANDLE_ERROR(cudaHostRegister(padded_kernel_->data(), padded_size_byte,
                                  cudaHostRegisterPortable));

    HANDLE_ERROR(cudaMemcpy(device_ports.at(image_id), padded_image_->data(),
                            padded_size_byte, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(device_ports.at(kernel_id), padded_kernel_->data(),
                            padded_size_byte, cudaMemcpyHostToDevice));


    multiviewnative::inplace_convolve_on_device<TransformT>(
        device_ports.at(image_id), device_ports.at(kernel_id),
        &PaddingT::extents_[0], device_memory_elements_required  
        );

    HANDLE_ERROR(cudaMemcpy(padded_image_->data(), device_ports.at(image_id),
                            padded_size_byte, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaHostUnregister(padded_image_->data()));
    HANDLE_ERROR(cudaHostUnregister(padded_kernel_->data()));

    this->padded_image_->resize(this->extents_);
    this->padded_kernel_->resize(this->extents_);

    // cut-out region of interest
    (*image_) = (*padded_image_)
        [boost::indices
             [range(this->offsets_[0], this->offsets_[0] + image_->shape()[0])]
             [range(this->offsets_[1], this->offsets_[1] + image_->shape()[1])]
             [range(this->offsets_[2],
                    this->offsets_[2] + image_->shape()[2])]];


  }

  /**
     \brief contract is equal to the one of inplace, except that this function
     expects the padded_kernel buffer is handed in through the function
     parameters (it is expected to have the same dimensions of the padded image,
     if not an exception is thrown)

     \param[in] _d_forwarded_padded_kernel forwarded kernel in the same data
     structure that is used internally (it is assumed _d_forwarded_padded_kernel has the right shape for a cufft inplace transform)

     \param[in] _d_image memory on device to be foreseen for the actual image data (it is assumed _d_image has the right shape for a cufft inplace transform)

     \return
     \retval

  */
  template <typename TransformT>
  void half_inplace(value_type* _d_forwarded_padded_kernel, 
		    value_type* _d_image, 
		    cudaStream_t* _forwarded_kernel_stream = 0,
		    cudaStream_t* _image_stream = 0
		    ) {
    // extend kernel and image according to inplace requirements
    // (docs.nvidia.com/cuda/cufft/index.html#multi-dimensional)

    
    unsigned long padded_size_byte =
        padded_image_->num_elements() * sizeof(value_type);

    //transfer image to device
    value_type* image_on_device = _d_image;
    cudaStream_t* image_tx = _image_stream;
    if(!_image_stream){
      image_tx = new cudaStream_t;
      HANDLE_ERROR(cudaStreamCreate(image_tx));
    }

    unsigned int flags = 0;


    if(cudaHostGetFlags(&flags,padded_image_->data())!=cudaSuccess){
      HANDLE_ERROR(cudaHostRegister(padded_image_->data(), padded_size_byte,
				    cudaHostRegisterPortable));
      cudaGetLastError();
    }
    HANDLE_ERROR(cudaPeekAtLastError());

    HANDLE_ERROR(cudaMemcpyAsync(// image_on_device
    				 _d_image, 
    				 padded_image_->data(),
    				 padded_size_byte, 
    				 cudaMemcpyHostToDevice,
    				 *image_tx));


    TransformT image_transform(image_on_device, &this->extents_[0]);
    image_transform.forward(image_tx);


    size_type transform_size =
      std::accumulate(this->extents_.begin(), this->extents_.end(), 1, std::multiplies<size_type>());
    size_type eff_fft_num_elements = padded_size_byte/(sizeof(value_type)*2);

    TransferT scale = 1.0 / TransferT(transform_size);

    unsigned numThreads = 128;
    unsigned numBlocks = largestDivisor(eff_fft_num_elements, numThreads);

    HANDLE_ERROR(cudaStreamSynchronize(*image_tx));
    HANDLE_ERROR(cudaStreamSynchronize(*_forwarded_kernel_stream));

    if(_forwarded_kernel_stream){
      modulateAndNormalize_kernel <<<numBlocks, numThreads,0,*_forwarded_kernel_stream>>>((cufftComplex*)image_on_device, 
    											  (cufftComplex*)_d_forwarded_padded_kernel,
    											  eff_fft_num_elements, 
    											  scale);
    }
    else{
      modulateAndNormalize_kernel <<<numBlocks, numThreads>>>((cufftComplex*)image_on_device, 
							      (cufftComplex*)_d_forwarded_padded_kernel,
							      eff_fft_num_elements, 
							      scale);
    }
    HANDLE_ERROR(cudaPeekAtLastError());

    image_transform.backward(image_tx);
  
    //get image back
    HANDLE_ERROR(cudaMemcpyAsync(padded_image_->data(), 
				 image_on_device,
				 padded_size_byte,
				 cudaMemcpyDeviceToHost,
				 *image_tx
				 ));
    HANDLE_ERROR(cudaStreamSynchronize(*image_tx));
    HANDLE_ERROR(cudaHostUnregister(padded_image_->data()));
    
    // this->padded_image_->resize(this->extents_);

    // // cut-out region of interest
    // (*image_) = (*padded_image_)
    //     [boost::indices
    //          [range(this->offsets_[0], this->offsets_[0] + image_->shape()[0])]
    //          [range(this->offsets_[1], this->offsets_[1] + image_->shape()[1])]
    //          [range(this->offsets_[2],
    //                 this->offsets_[2] + image_->shape()[2])]];

    if(!_image_stream){
      HANDLE_ERROR(cudaStreamDestroy(*image_tx));
      delete image_tx;
    }

  }
  

  ~gpu_convolve() {
    if(image_)
      delete image_;
    if(kernel_)
      delete kernel_;
    if(padded_image_)
      delete padded_image_;
    if(padded_kernel_)
      delete padded_kernel_;

    // for (int i = 0; i < 2; ++i)
    // 	HANDLE_ERROR(cudaStreamDestroy(streams_[i]));
  }

  void set_device(const int& _device) {
    HANDLE_ERROR(cudaSetDevice(_device));
  };

 private:
  

  stack_ref_t* image_;
  stack_t* padded_image_;

  stack_ref_t* kernel_;
  stack_t* padded_kernel_;

  std::vector<unsigned> cufft_shape_;
};
}

#endif /* _GPU_CONVOLVE_H_ */

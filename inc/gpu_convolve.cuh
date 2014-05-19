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

  typedef multiviewnative::stack_on_device<multiviewnative::image_stack, multiviewnative::asynch> asynch_stack_on_device;
  typedef multiviewnative::stack_on_device<multiviewnative::image_stack, multiviewnative::synch> synch_stack_on_device;
  
  template <typename TransformT, typename TransferT, typename DimT>
  void inplace_asynch_convolve_on_device(TransferT* _image_on_device, 
				  TransferT* _kernel_on_device, 
				  DimT*   _extents,
				  const unsigned& _fft_num_elements,
				  const std::vector<cudaStream_t*>& _streams){
            

    TransformT image_transform(_image_on_device, _extents);
    TransformT kernel_transform(_kernel_on_device, _extents);
    image_transform.forward(_streams[0]);
    kernel_transform.forward(_streams[1]);

    DimT transform_size = std::accumulate(_extents,_extents + 3,1,std::multiplies<DimT>());      
    unsigned eff_fft_num_elements = _fft_num_elements/2;

    TransferT scale = 1.0 / (transform_size);

    unsigned numThreads = 128;
    unsigned numBlocks = largestDivisor(eff_fft_num_elements,numThreads);

    HANDLE_ERROR(cudaStreamSynchronize(*_streams[1]));
    HANDLE_ERROR(cudaStreamSynchronize(*_streams[0]));    
    modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)_image_on_device, 
							  (cufftComplex *)_kernel_on_device, 
							  eff_fft_num_elements, 
							  scale);
    HANDLE_ERROR(cudaPeekAtLastError());
      
    image_transform.backward(_streams[0]);

  }

  template <typename TransformT, typename TransferT, typename DimT>
  void inplace_convolve_on_device(TransferT* _image_on_device, 
				  TransferT* _kernel_on_device, 
				  DimT*   _extents,
				  const unsigned& _fft_num_elements){
            

    TransformT image_transform(_image_on_device, _extents);
    TransformT kernel_transform(_kernel_on_device, _extents);
    image_transform.forward();
    kernel_transform.forward();

    DimT transform_size = std::accumulate(_extents,_extents + 3,1,std::multiplies<DimT>());      
    unsigned eff_fft_num_elements = _fft_num_elements/2;

    TransferT scale = 1.0 / (transform_size);

    unsigned numThreads = 128;
    unsigned numBlocks = largestDivisor(eff_fft_num_elements,numThreads);

    
    modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)_image_on_device, 
							  (cufftComplex *)_kernel_on_device, 
							  eff_fft_num_elements, 
							  scale);
    HANDLE_ERROR(cudaPeekAtLastError());
      
    image_transform.backward();
  }

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

    
    template <typename TransformT>
    void inplace(){

      //extend kernel and image according to inplace requirements (docs.nvidia.com/cuda/cufft/index.html#multi-dimensional)
      std::vector<unsigned> inplace_extents(this->extents_.size());
      adapt_extents_for_fftw_inplace(padded_image_->storage_order(),this->extents_, inplace_extents);
      size_type device_memory_elements_required = std::accumulate(inplace_extents.begin(),inplace_extents.end(),1,std::multiplies<size_type>());    
       
	//place image and kernel on device (add extra space on device for cufft processing)
      // asynch_stack_on_device image_on_device(padded_image_,device_memory_elements_required);
      // asynch_stack_on_device kernel_on_device(padded_kernel_,device_memory_elements_required);
	 
      // for (int i = 0; i < 2; ++i)
      // 	HANDLE_ERROR(cudaStreamCreate(&streams_[i]));

      // image_on_device.push_to_device(&streams_[0]);
      // kernel_on_device.push_to_device(&streams_[1]);
      unsigned long padded_size_byte = padded_image_->num_elements()*sizeof(value_type);
      unsigned long inplace_size_byte = device_memory_elements_required *sizeof(value_type);
      
      multiviewnative::device_memory_ports<value_type,2> device_ports;
      device_ports.create_all_ports(inplace_size_byte);
      static const int image_id = 0;
      static const int kernel_id = 1;

      device_ports.template add_stream_for<image_id>();
      device_ports.add_stream_for(kernel_id);

      HANDLE_ERROR( cudaHostRegister(padded_image_->data()   , padded_size_byte , cudaHostRegisterPortable) );
      HANDLE_ERROR( cudaHostRegister(padded_kernel_->data()   , padded_size_byte , cudaHostRegisterPortable) );

      HANDLE_ERROR( cudaMemcpy(device_ports.at(image_id ), padded_image_->data()   , padded_size_byte , cudaMemcpyHostToDevice) );
      HANDLE_ERROR( cudaMemcpy(device_ports.at(kernel_id ), padded_kernel_->data()   , padded_size_byte , cudaMemcpyHostToDevice) );
      
      // device_ports.template send<image_id >(  padded_image_->data() ,  padded_size_byte);
      // device_ports.template send<kernel_id>(  padded_kernel_->data() , padded_size_byte);
      // std::vector<cudaStream_t*> wave(2);
      // device_ports.template streams_of_two<image_id,kernel_id>(wave);

      multiviewnative::inplace_convolve_on_device<TransformT>(device_ports.at(image_id ) ,
							      device_ports.at(kernel_id) ,
							      &PaddingT::extents_[0] ,
							      device_memory_elements_required // , 
							      // wave
							      );

      // device_ports.template sync_stream<0>();
      // device_ports.template receive<0>(  padded_image_->data() , padded_size_byte );
      HANDLE_ERROR( cudaMemcpy(padded_image_->data()   , device_ports.at(image_id ), padded_size_byte , cudaMemcpyDeviceToHost) );
      
      
      //cut-out region of interest
      (*image_) = (*padded_image_)[ boost::indices[range(this->offsets_[0], this->offsets_[0]+image_->shape()[0])][range(this->offsets_[1], this->offsets_[1]+image_->shape()[1])][range(this->offsets_[2], this->offsets_[2]+image_->shape()[2])] ];
      
      HANDLE_ERROR( cudaHostUnregister(padded_image_->data()) );
      HANDLE_ERROR( cudaHostUnregister(padded_kernel_->data()) );

    }

    ~gpu_convolve(){
      delete image_;
      delete kernel_;
      delete padded_image_;
      delete padded_kernel_;

      // for (int i = 0; i < 2; ++i)
      // 	HANDLE_ERROR(cudaStreamDestroy(streams_[i]));

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




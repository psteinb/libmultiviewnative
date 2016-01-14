#ifndef _CUFFT_UTILS_H_
#define _CUFFT_UTILS_H_
#include <vector>
#include <algorithm>

#include "cufft.h"

#include "point.h"
#include "cuda_helpers.cuh"

#include "cufft_interface.cuh"
#include "plan_store.cuh"

#define HANDLE_CUFFT_ERROR(err) (multiviewnative::gpu::HandleCufftError(err, __FILE__, __LINE__))

namespace multiviewnative {


  template <typename TransferT,
	    cufftCompatibility Mode = CUFFT_COMPATIBILITY_NATIVE>
  class inplace_3d_transform_on_device {



  public:
    typedef gpu::cufft_api<TransferT> api; 
    typedef typename api::real_t real_type;
    typedef typename api::complex_t complex_type;
    typedef typename api::plan_t plan_type;
    typedef gpu::plan_store<real_type> plan_store;

    typedef long size_type;

    static const int dimensionality = 3;

    template <typename DimT>
    inplace_3d_transform_on_device(TransferT* _input, DimT* _shape)
      : input_(_input), shape_(_shape, _shape + dimensionality) {
    }

    void forward(cudaStream_t* _stream = 0) {

    
      if(!plan_store::get()->has_key(shape_))
	plan_store::get()->add(shape_);
    
      plan_type* plan = plan_store::get()->get_forward(shape_);

      if (_stream) 
	HANDLE_CUFFT_ERROR(cufftSetStream(*plan, *_stream));
      else
	HANDLE_ERROR(cudaDeviceSynchronize());

      HANDLE_CUFFT_ERROR(
			 cufftExecR2C(*plan, input_, (complex_type*)input_));


    };

    void backward(cudaStream_t* _stream = 0) {

      if(!plan_store::get()->has_key(shape_))
	plan_store::get()->add(shape_);
    
      plan_type* plan = plan_store::get()->get_backward(shape_);

      if (_stream) 
	HANDLE_CUFFT_ERROR(cufftSetStream(*plan, *_stream));
      else
	HANDLE_ERROR(cudaDeviceSynchronize());

      HANDLE_CUFFT_ERROR(
			 cufftExecC2R(*plan, (complex_type*)input_, input_));

    };

    ~inplace_3d_transform_on_device() {};

  private:
    TransferT* input_;
    multiviewnative::shape_t shape_;


  };

  namespace gpu {
    //loosely based on nvidia-samples/6_Advanced/concurrentKernels/concurrentKernels.cu
    template <typename stack_type>
    void batched_fft_async2plans(std::vector<stack_type>& _prepared_stacks,
				 const multiviewnative::shape_t& _transform_shape,
				 std::vector<float*>& _src_buffers,
				 bool register_input_stacks = true) {

      typedef plan_store<float> plan_store;

      //create 2! plans to satisfy 2 streams
      if(!plan_store::get()->has_key(_transform_shape))
	plan_store::get()->add(_transform_shape);

      std::vector<cufftHandle*> _plans(2, 0);
      _plans[0] = plan_store::get()->get_forward(_transform_shape);
      
      if(_plans.size()>1){
	_plans[1] = new cufftHandle;
	HANDLE_CUFFT_ERROR(cufftPlan3d(_plans[1],                 //
				       (int)_transform_shape[0], //
				       (int)_transform_shape[1], //
				       (int)_transform_shape[2], //
				       CUFFT_R2C)                    //
			   );
      }
      
      //create streams
      std::vector<cudaStream_t*> streams(_plans.size());
      for( unsigned count = 0;count < streams.size();++count ){
	streams[count] = new cudaStream_t;
	HANDLE_ERROR(cudaStreamCreate(streams[count]));
      }

      unsigned stack_size_in_byte = _prepared_stacks[0].num_elements() * sizeof(float);
      if(register_input_stacks){
	for( unsigned count = 0;count < _prepared_stacks.size();++count ){
	  HANDLE_ERROR(cudaHostRegister((void*)_prepared_stacks[count].data(), 
					stack_size_in_byte,
					cudaHostRegisterPortable));
  
	}
      }

      std::vector<cudaEvent_t> before_plan_execution(_prepared_stacks.size());
      for( cudaEvent_t& e : before_plan_execution){
	HANDLE_ERROR(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      }
    
      float* d_buffer = 0;
      
      unsigned modulus_index = 0;
      for( unsigned count = 0;count < _prepared_stacks.size();++count ){

	modulus_index = count % streams.size();
	d_buffer = _src_buffers[count % streams.size()];
    
	HANDLE_ERROR(cudaMemcpyAsync(d_buffer,
				     _prepared_stacks[count].data(), 
				     stack_size_in_byte,
				     cudaMemcpyHostToDevice,
				     *streams[modulus_index]
				     ));

    
	HANDLE_CUFFT_ERROR(cufftSetStream(*_plans[modulus_index],				 
					  *streams[modulus_index] )
			   );

	HANDLE_ERROR(cudaEventRecord(before_plan_execution[count],*streams[modulus_index]));
	if(count>0)
	  HANDLE_ERROR(cudaStreamWaitEvent(*streams[modulus_index], before_plan_execution[count-1],0));
    
	HANDLE_CUFFT_ERROR(
			   cufftExecR2C(*_plans[modulus_index], d_buffer, (cufftComplex*)d_buffer));
    
				 
	HANDLE_ERROR(cudaMemcpyAsync(_prepared_stacks[count].data(), 
				     d_buffer,
				     stack_size_in_byte,
				     cudaMemcpyDeviceToHost,
				     *streams[modulus_index])
		     );
    
      }
   
      //clean-up
      for (unsigned count = 0;count < streams.size();++count){
	HANDLE_ERROR(cudaStreamSynchronize(*streams[count]));
	HANDLE_ERROR(cudaStreamDestroy(*streams[count]));
      }
  
      for (unsigned count = 0;count < _prepared_stacks.size();++count){
	HANDLE_ERROR(cudaEventDestroy(before_plan_execution[count]));
	if(register_input_stacks)
	  HANDLE_ERROR(cudaHostUnregister((void*)_prepared_stacks[count].data()));
      }
  
      HANDLE_CUFFT_ERROR(cufftDestroy(*_plans.back()));
      delete _plans.back();
      
    }
  } // gpu

}
#endif /* _FFT_UTILS_H_ */

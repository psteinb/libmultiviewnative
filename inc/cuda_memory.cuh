#ifndef _CUDA_MEMORY_H_
#define _CUDA_MEMORY_H_

#include "cuda_helpers.cuh"

namespace multiviewnative {

  template <typename ImageStackT>
  struct stack_on_device {

    typedef typename ImageStackT::element value_type;
    
    ImageStackT* host_stack_;
    value_type* device_stack_;
    unsigned size_in_byte_;

    stack_on_device():
      host_stack_(0),
      device_stack_(0),
      size_in_byte_(0)
    {}
    
    stack_on_device( ImageStackT& _other):
      host_stack_(&_other),
      device_stack_(0),
      size_in_byte_(sizeof(value_type)*_other.num_elements())
    {
      HANDLE_ERROR( cudaMalloc( (void**)&(device_stack_), size_in_byte_ ) );
      HANDLE_ERROR( cudaHostRegister( host_stack_->data(), size_in_byte_ , cudaHostRegisterPortable) );
    }

    stack_on_device(ImageStackT* _other):
      host_stack_(_other),
      device_stack_(0),
      size_in_byte_(sizeof(value_type)*_other->num_elements())
    {
      HANDLE_ERROR( cudaMalloc( (void**)&(device_stack_), size_in_byte_ ) );
      HANDLE_ERROR( cudaHostRegister( host_stack_->data(), size_in_byte_ , cudaHostRegisterPortable) );
    }


    stack_on_device& operator=(ImageStackT& _rhs){
      this->host_stack_ = &_rhs;

      size_in_byte_ = sizeof(value_type)*_rhs.num_elements();

      HANDLE_ERROR( cudaHostRegister( host_stack_->data(), size_in_byte_ , cudaHostRegisterPortable) );

      if(_rhs.num_elements()!=(size_in_byte_/sizeof(value_type))){
	this->clear();
	HANDLE_ERROR( cudaMalloc( (void**)&(device_stack_), size_in_byte_ ) );
      }
      
      return *this;
    }
    
    void pull(cudaStream_t* _stream = 0){
      if(_stream)
	async_pull(_stream);
      else
	sync_pull();
    }

    void async_pull(cudaStream_t* _stream = 0){
      HANDLE_ERROR( cudaMemcpyAsync(host_stack_->data(), device_stack_ , 
				    size_in_byte_ , 
				    cudaMemcpyDeviceToHost, *_stream ) );
    }

    void sync_pull(){
      HANDLE_ERROR( cudaMemcpy(host_stack_->data(), device_stack_ , 
			       size_in_byte_ , 
			       cudaMemcpyDeviceToHost ) );
    }

    void push(cudaStream_t* _stream = 0){

      if(_stream)
	async_push(_stream);
      else
	sync_push();
      
    }

    void async_push(cudaStream_t* _stream = 0){
    
      HANDLE_ERROR( cudaMemcpyAsync(device_stack_ , host_stack_->data(), 
				    size_in_byte_ , 
				    cudaMemcpyHostToDevice, *_stream ) );
    
      
    }

    void sync_push(){
    
      HANDLE_ERROR( cudaMemcpy(device_stack_ , host_stack_->data(), 
			       size_in_byte_ , 
			       cudaMemcpyHostToDevice ) );
    
      
    }

    void clear(){
      HANDLE_ERROR( cudaHostUnregister( host_stack_->data()));
      if(device_stack_)
	HANDLE_ERROR( cudaFree( device_stack_ ) );
    }
    
    ~stack_on_device(){
      this->clear();
    }
  };

};


#endif /* _CUDA_MEMORY_H_ */


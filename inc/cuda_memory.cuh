#ifndef _CUDA_MEMORY_H_
#define _CUDA_MEMORY_H_

#include "cuda_helpers.cuh"

namespace multiviewnative {

  template <typename ImageStackT> 
  struct synch {

    typedef typename ImageStackT::element value_type;
    
    void push(ImageStackT* _host_ptr, value_type* _device_ptr, cudaStream_t* _stream = 0 ){
      HANDLE_ERROR( cudaMemcpy(_device_ptr , _host_ptr->data(),  _host_ptr->num_elements()*sizeof(value_type) , cudaMemcpyHostToDevice ) );
    }

    void pull(value_type* _device_ptr, ImageStackT* _host_ptr, cudaStream_t* _stream = 0 ){
      HANDLE_ERROR( cudaMemcpy(_host_ptr->data(), _device_ptr , _host_ptr->num_elements()*sizeof(value_type) , cudaMemcpyDeviceToHost ) );
    }
  };

  template <typename ImageStackT>     
  struct asynch {

    typedef typename ImageStackT::element value_type;

    void push(const ImageStackT* _host_ptr, 
	      value_type* _device_ptr, 
	      cudaStream_t* _stream = 0 ){
      HANDLE_ERROR( cudaMemcpyAsync(_device_ptr , _host_ptr->data(),  _host_ptr->num_elements()*sizeof(value_type) , cudaMemcpyHostToDevice, *_stream ) );
    }

    
    void pull(value_type* _device_ptr, ImageStackT* _host_ptr, cudaStream_t* _stream = 0 ){
      HANDLE_ERROR( cudaMemcpyAsync(_host_ptr->data(), _device_ptr , _host_ptr->num_elements()*sizeof(value_type) , cudaMemcpyDeviceToHost, *_stream ) );
    }
  };

  template <typename ImageStackT, template <typename> class IOPolicy >
  struct stack_on_device : public IOPolicy<ImageStackT> {

    typedef typename ImageStackT::element value_type;
    typedef IOPolicy<ImageStackT> io_policy;

    ImageStackT* host_stack_;
    value_type* device_stack_ptr_;
    unsigned size_in_byte_;

    stack_on_device():
      host_stack_(0),
      device_stack_ptr_(0),
      size_in_byte_(0)
    {}
    
    stack_on_device( ImageStackT& _other, const size_t& _num_elements = 0):
      host_stack_(&_other),
      device_stack_ptr_(0),
      size_in_byte_(sizeof(value_type)*((_num_elements) ? _num_elements : _other.num_elements()))
    {
      HANDLE_ERROR( cudaMalloc( (void**)&(device_stack_ptr_), size_in_byte_ ) );
      HANDLE_ERROR( cudaHostRegister( host_stack_->data(), sizeof(value_type)*_other.num_elements() , cudaHostRegisterPortable) );
    }

    stack_on_device(ImageStackT* _other, const size_t& _num_elements = 0):
      host_stack_(_other),
      device_stack_ptr_(0),
      size_in_byte_(sizeof(value_type)*((_num_elements) ? _num_elements : _other->num_elements()))
    {
      HANDLE_ERROR( cudaMalloc( (void**)&(device_stack_ptr_), size_in_byte_ ) );
      HANDLE_ERROR( cudaHostRegister( host_stack_->data(), sizeof(value_type)*_other->num_elements() , cudaHostRegisterPortable) );
    }


    stack_on_device& operator=(ImageStackT& _rhs){
      this->host_stack_ = &_rhs;

      size_in_byte_ = sizeof(value_type)*_rhs.num_elements();

      HANDLE_ERROR( cudaHostRegister( host_stack_->data(), size_in_byte_ , cudaHostRegisterPortable) );

      if(_rhs.num_elements()!=(size_in_byte_/sizeof(value_type))){
	this->clear();
	HANDLE_ERROR( cudaMalloc( (void**)&(device_stack_ptr_), size_in_byte_ ) );
      }
      
      return *this;
    }
    
    void resize_device_memory(const size_t& _num_elements){
      if(device_stack_ptr_){
	HANDLE_ERROR( cudaFree( device_stack_ptr_ ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(device_stack_ptr_), _num_elements*sizeof(value_type) ) );
      }
    }
    
    void pull_from_device(cudaStream_t* _stream = 0){
      io_policy::pull(device_stack_ptr_, host_stack_,  _stream);
    }
    
    void push_to_device(cudaStream_t* _stream = 0) {
      io_policy::push(host_stack_, device_stack_ptr_, _stream);
    }

    void clear(){

      if(host_stack_)
	HANDLE_ERROR( cudaHostUnregister( host_stack_->data()));

      if(device_stack_ptr_)
	HANDLE_ERROR( cudaFree( device_stack_ptr_ ) );

    }
    
    ~stack_on_device(){
      this->clear();
    }
  };

};


#endif /* _CUDA_MEMORY_H_ */




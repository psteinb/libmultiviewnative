#ifndef _CUDA_MEMORY_H_
#define _CUDA_MEMORY_H_

#include "cuda_helpers.cuh"
#include "boost/static_assert.hpp"

namespace multiviewnative {

  template <typename ValueT> 
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

  enum      port_id  {
    psi       =        0,
    view      =        1,
    kernel1   =        2,
    kernel2   =        3,
    weights   =        4,
    integral  =        5
  };



  template<typename ValueT, unsigned num_items = 4>
  struct device_memory_ports {
    
    typedef ValueT value_type;

    std::vector< value_type* > device_ptr_;
    std::vector< size_t > device_size_in_byte_;
    std::vector< cudaStream_t > device_streams_;

    device_memory_ports():
      device_ptr_(num_items),
      device_size_in_byte_(num_items),
      device_streams_(num_items){
      
      for (int i = 0; i < num_items; ++i){
	device_ptr_[i] = 0;
	device_size_in_byte_[i] = 0;
      	HANDLE_ERROR(cudaStreamCreate(&device_streams[i]));
      }
      
    }

    template<typename AsynchImageStackT>
    device_memory_ports(AsynchImageStackT* _stack_array):
      device_ptr_(num_items),
      device_size_in_byte_(num_items),
      device_streams_(num_items){
      
      for (int i = 0; i < num_items; ++i){
      	HANDLE_ERROR(cudaStreamCreate(&device_streams[i]));
	device_ptr_[i] = _stack_array[i].device_stack_ptr_;
	device_size_in_byte_[i] = _stack_array[i].size_in_byte_;
      }
      
    }

    ~device_memory_ports(){
      for (int i = 0; i < num_items; ++i){
	HANDLE_ERROR(cudaStreamDestroy(device_streams[i]));
	if(device_ptr_[num])
	  HANDLE_ERROR( cudaFree( device_ptr_[num] ) );
      }
    }

    template<unsigned num>
    void setup(const size_t& _size_on_dev ){
      boost::static_assert(num < num_items, "device_memory_ports::setup \t trying to setup memory region that is not under control");
      device_size_in_byte_[num] = _size_on_dev;
      HANDLE_ERROR( cudaMalloc( (void**)&(device_ptr_[num]), device_size_in_byte_[num] ) );

    }

    
    void setup_all(const size_t& _size_on_dev ){
      
      for (int i = 0; i < num_items; ++i){
	device_size_in_byte_[num] = _size_on_dev;
	if(device_ptr_[num])
	  HANDLE_ERROR( cudaFree( device_ptr_[num] ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(device_ptr_[num]), device_size_in_byte_[num] ) );
      }

    }

    template<unsigned num>
    void onto_device(value_type* _host_ptr){
      boost::static_assert(num < num_items, "device_memory_ports::onto_device(value_type*) \t trying to setup memory region that is not under control");
      if(device_ptr_[num])
	HANDLE_ERROR( cudaMemcpyAsync(device_ptr_[num] , 
				      _host_ptr, 
				      device_size_in_byte_[num]  , 
				      cudaMemcpyHostToDevice, 
				      device_streams[num] ) );
      else
	std::cerr << "device_memory_ports::onto_device(value_type*)\t device pointer " << num << " unitialized (nothing to transfer)\n" ;
    }

    template<unsigned num, typename ImageStackT>
    void onto_device(const ImageStackT* _stack){
      boost::static_assert(num < num_items, "device_memory_ports::onto_device(const ImageStackT*) \t trying to setup memory region that is not under control");
      if(device_ptr_[num])
	HANDLE_ERROR( cudaMemcpyAsync(device_ptr_[num] , 
				      _stack->data(), 
				      device_size_in_byte_[num]  , 
				      cudaMemcpyHostToDevice, 
				      device_streams[num] ) );
      else
	std::cerr << "device_memory_ports::onto_device(const ImageStackT*)\t device pointer " << num << " unitialized (nothing to transfer)\n" ;
    }

    template<unsigned first, unsigned second>
    std::vector<cudaStream_t*> streams_of(){
      boost::static_assert(first < num_items && second && num_items, "device_memory_ports::streams_of(const ImageStackT*) \t trying to setup memory region that is not under control");
      std::vector<cudaStream_t*> temp(2);
      temp[0] = &device_streams[first];
      temp[1] = &device_streams[second];
      
      return temp;
    }

    template<unsigned num>
    value_type* at(){
      boost::static_assert(num < num_items, "device_memory_ports::at() \t trying to access device pointer that does not exist");
      return device_ptr_[num];
    }
  };

};


#endif /* _CUDA_MEMORY_H_ */




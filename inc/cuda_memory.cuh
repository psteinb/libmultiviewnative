#ifndef _CUDA_MEMORY_H_
#define _CUDA_MEMORY_H_

#include <map>

#include "cuda_helpers.cuh"
#include "boost/static_assert.hpp"

namespace multiviewnative {

template <typename ImageStackT>
struct synch {

  typedef typename ImageStackT::element value_type;

  void push(const ImageStackT* _host_ptr, value_type* _device_ptr,
            cudaStream_t* _stream = 0) {
    HANDLE_ERROR(cudaMemcpy(_device_ptr, _host_ptr->data(),
                            _host_ptr->num_elements() * sizeof(value_type),
                            cudaMemcpyHostToDevice));
  }

  void pull(value_type* _device_ptr, ImageStackT* _host_ptr,
            cudaStream_t* _stream = 0) {
    HANDLE_ERROR(cudaMemcpy(_host_ptr->data(), _device_ptr,
                            _host_ptr->num_elements() * sizeof(value_type),
                            cudaMemcpyDeviceToHost));
  }
};

template <typename ImageStackT>
struct asynch {

  typedef typename ImageStackT::element value_type;

  void push(const ImageStackT* _host_ptr, value_type* _device_ptr,
            cudaStream_t* _stream = 0) {
    HANDLE_ERROR(cudaMemcpyAsync(_device_ptr, _host_ptr->data(),
                                 _host_ptr->num_elements() * sizeof(value_type),
                                 cudaMemcpyHostToDevice, *_stream));
  }

  void pull(value_type* _device_ptr, ImageStackT* _host_ptr,
            cudaStream_t* _stream = 0) {
    HANDLE_ERROR(cudaMemcpyAsync(_host_ptr->data(), _device_ptr,
                                 _host_ptr->num_elements() * sizeof(value_type),
                                 cudaMemcpyDeviceToHost, *_stream));
  }
};

template <typename T>
unsigned obtain_cuda_host_mem_flags(T* _ptr) {
  unsigned value = 0;
  cudaError_t errcode = cudaHostGetFlags(&value, (void*)_ptr);
  if (errcode != cudaSuccess) {
    // clearing the cuda driver error state
    // see
    // http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDART__ERROR_g0a933755a73d874d0d7051a3eb2aa533.html#g0a933755a73d874d0d7051a3eb2aa533
    cudaGetLastError();
    return 0;
  } else
    return value;
}

// TODO: refactor this class
template <typename ImageStackT, template <typename> class IOPolicy = synch>
struct stack_on_device : public IOPolicy<ImageStackT> {

  typedef typename ImageStackT::element value_type;
  typedef IOPolicy<ImageStackT> io_policy;

  // ownership taken
  value_type* device_stack_ptr_;
  unsigned long size_in_byte_;

  //////////////////////////////////////////////////////////////////////////////////////////
  // Constructors

  stack_on_device() : device_stack_ptr_(0), size_in_byte_(0) {}

  stack_on_device(const size_t& _num_elements)
      : device_stack_ptr_(0),
        size_in_byte_(sizeof(value_type) * (_num_elements)) {
    HANDLE_ERROR(cudaMalloc((void**)&(device_stack_ptr_), size_in_byte_));
  }

  // RAII
  stack_on_device(const ImageStackT& _other, const size_t& _num_elements = 0,
                  cudaStream_t* _stream = 0)
      : device_stack_ptr_(0),
        size_in_byte_(sizeof(value_type) * ((_num_elements)
                                                ? _num_elements
                                                : _other.num_elements())) {
    HANDLE_ERROR(cudaMalloc((void**)&(device_stack_ptr_), size_in_byte_));

    push_to_device(_other, _stream);
  }

  stack_on_device(ImageStackT* _other, const size_t& _num_elements = 0,
                  cudaStream_t* _stream = 0)
      : device_stack_ptr_(0),
        size_in_byte_(sizeof(value_type) * ((_num_elements)
                                                ? _num_elements
                                                : _other->num_elements())) {
    HANDLE_ERROR(cudaMalloc((void**)&(device_stack_ptr_), size_in_byte_));
    push_to_device(*_other, _stream);
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // Copy
  stack_on_device(const stack_on_device& _other)
      : device_stack_ptr_(0), size_in_byte_(_other.size_in_byte_) {

    HANDLE_ERROR(cudaMalloc((void**)&(device_stack_ptr_), size_in_byte_));

    if(_other.device_stack_ptr_)
      HANDLE_ERROR(cudaMemcpy(device_stack_ptr_, _other.device_stack_ptr_,
			      size_in_byte_, cudaMemcpyDeviceToDevice));
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // Assignment
  stack_on_device& operator=(const ImageStackT& _rhs) {

    stack_on_device temp(_rhs);
    swap(*this, _rhs);

    return *this;
  }

  stack_on_device& operator=(stack_on_device _rhs) {

    swap(*this, _rhs);
    return *this;
  }

  //////////////////////////////////////////////////////////////////////////////////////////
  // member functions
  friend void swap(stack_on_device& _first, stack_on_device& _second) {
    std::swap(_first.device_stack_ptr_, _second.device_stack_ptr_);
    std::swap(_first.size_in_byte_, _second.size_in_byte_);
  }

  void resize_device_memory(const size_t& _num_elements) {
    if (device_stack_ptr_) {
      HANDLE_ERROR(cudaFree(device_stack_ptr_));
      HANDLE_ERROR(cudaMalloc((void**)&(device_stack_ptr_),
                              _num_elements * sizeof(value_type)));
    }
  }

  void pull_from_device(ImageStackT& _other, cudaStream_t* _stream = 0) {

    if (_other.num_elements() * sizeof(value_type) <= size_in_byte_) {
      io_policy::pull(device_stack_ptr_, &_other, _stream);
    } else {
      std::stringstream msg("");
      msg << "[mvn::stack_on_device]\tunable to pull cpu based image stack "
             "from device (cpu size: "
          << _other.num_elements() * sizeof(value_type) / float(1 << 20)
          << " MB,allocated gpu device memory: "
          << size_in_byte_ / float(1 << 20) << " MB)\n";
      throw std::runtime_error(msg.str());
    }
  }

  void push_to_device(const ImageStackT& _other, cudaStream_t* _stream = 0) {

    if (_other.num_elements() * sizeof(value_type) <= size_in_byte_)
      io_policy::push(&_other, device_stack_ptr_, _stream);
    else {
      std::stringstream msg("");
      msg << "[mvn::stack_on_device]\tunable to push cpu based image stack to "
             "device (cpu size: "
          << _other.num_elements() * sizeof(value_type) / float(1 << 20)
          << " MB,allocated gpu device memory: "
          << size_in_byte_ / float(1 << 20) << " MB)\n";
      throw std::runtime_error(msg.str());
    }
  }

  void clear() {

    if (device_stack_ptr_) {
      HANDLE_ERROR(cudaFree(device_stack_ptr_));
      device_stack_ptr_ = 0;
      size_in_byte_ = 0;
    }
  }

  ~stack_on_device() { this->clear(); }

  // const value_type* data() const {
  //   return device_stack_ptr_;
  // }

  value_type* data() { return device_stack_ptr_; }
};

enum port_id {
  psi = 0,
  view = 1,
  kernel1 = 2,
  kernel2 = 3,
  weights = 4,
  integral = 5
};

template <typename ValueT, int num_ports = 4>
struct device_memory_ports {

  typedef ValueT value_type;
  typedef std::map<int, cudaStream_t> stream_map;

  std::vector<value_type*> device_ptr_;
  std::vector<unsigned long> device_size_in_byte_;
  stream_map ports_;

  device_memory_ports()
      : device_ptr_(num_ports), device_size_in_byte_(num_ports), ports_() {

    for (int i = 0; i < num_ports; ++i) {
      device_ptr_[i] = 0;
      device_size_in_byte_[i] = 0;
    }
  }

  ~device_memory_ports() {

    stream_map::iterator pbegin = ports_.begin();
    stream_map::iterator pend = ports_.end();

    for (; pbegin != pend; ++pbegin)
      HANDLE_ERROR(cudaStreamDestroy(pbegin->second));

    for (int p = 0; p < num_ports; ++p) {
      if (device_ptr_[p]) HANDLE_ERROR(cudaFree(device_ptr_[p]));
    }
  }

  template <int port_index>
  void create_port(const unsigned long& _size_on_dev) {
    BOOST_STATIC_ASSERT_MSG(port_index < num_ports,
                            "device_memory_ports::create_port \t trying to "
                            "setup memory port that is not under control");
    device_size_in_byte_[port_index] = _size_on_dev;
    HANDLE_ERROR(cudaMalloc((void**)&(device_ptr_[port_index]),
                            device_size_in_byte_[port_index]));
  }

  void create_port(const int& _port_index, const unsigned long& _size_on_dev) {

    if (_port_index < num_ports) {
      device_size_in_byte_[_port_index] = _size_on_dev;
      HANDLE_ERROR(cudaMalloc((void**)&(device_ptr_[_port_index]),
                              device_size_in_byte_[_port_index]));
    } else {
      std::cerr << "device_memory_ports::create_port \t port_index "
                << _port_index << " out of bounds (max: " << num_ports << ")\n";
    }
  }

  void create_all_ports(const unsigned long& _size_on_dev) {

    for (int p = 0; p < num_ports; ++p) {
      device_size_in_byte_[p] = _size_on_dev;
      HANDLE_ERROR(
          cudaMalloc((void**)&(device_ptr_[p]), device_size_in_byte_[p]));
    }
  }

  template <int port_index>
  void add_stream_for() {
    BOOST_STATIC_ASSERT_MSG(port_index < num_ports,
                            "device_memory_ports::add_stream_for \t trying to "
                            "setup memory port that is not under control");
    cudaStream_t temp;
    HANDLE_ERROR(cudaStreamCreate(&temp));
    ports_[port_index] = temp;
  }

  void add_stream_for(const int& _port_index) {

    if (_port_index < num_ports) {
      cudaStream_t temp;
      HANDLE_ERROR(cudaStreamCreate(&temp));
      ports_[_port_index] = temp;
    } else {
      std::cerr << "device_memory_ports::add_stream_for \t port_index "
                << _port_index << " out of bounds (max: " << num_ports << ")\n";
    }
  }

  template <int port_index>
  void send(value_type* _host_ptr, unsigned long _size_in_byte = 0) {
    if (!_size_in_byte) _size_in_byte = device_size_in_byte_[port_index];

    if (_size_in_byte > device_size_in_byte_[port_index]) {
      std::cerr << "device_memory_ports::send(value_type*, unsigned long)\t "
                   "size given exceeds allocated size (given: " << _size_in_byte
                << " B, allocated: " << device_size_in_byte_[port_index]
                << " B)\n";
      _size_in_byte = device_size_in_byte_[port_index];
    }

    BOOST_STATIC_ASSERT_MSG(port_index < num_ports,
                            "device_memory_ports::send(value_type*) \t trying "
                            "to setup memory region that is not under control");
    if (device_ptr_[port_index]) {
      if (ports_.count(port_index))
        HANDLE_ERROR(cudaMemcpyAsync(device_ptr_[port_index], _host_ptr,
                                     _size_in_byte, cudaMemcpyHostToDevice,
                                     ports_[port_index]));
      else
        HANDLE_ERROR(cudaMemcpy(device_ptr_[port_index], _host_ptr,
                                _size_in_byte, cudaMemcpyHostToDevice));

    } else
      std::cerr << "device_memory_ports::send(value_type*)\t device pointer "
                << port_index << " unitialized (nothing to transfer)\n";
  }

  void send(const int& _port_index, value_type* _host_ptr,
            unsigned long _size_in_byte = 0) {
    if (!_size_in_byte) _size_in_byte = device_size_in_byte_[_port_index];

    if (_size_in_byte > device_size_in_byte_[_port_index]) {
      std::cerr << "device_memory_ports::send(const int&, value_type*, "
                   "unsigned long)\t size given exceeds allocated size (given: "
                << _size_in_byte
                << " B, allocated: " << device_size_in_byte_[_port_index]
                << " B)\n";
      _size_in_byte = device_size_in_byte_[_port_index];
    }

    if (_port_index < num_ports) {
      if (device_ptr_[_port_index]) {
        if (ports_.count(_port_index))
          HANDLE_ERROR(cudaMemcpyAsync(device_ptr_[_port_index], _host_ptr,
                                       _size_in_byte, cudaMemcpyHostToDevice,
                                       ports_[_port_index]));
        else
          HANDLE_ERROR(cudaMemcpy(device_ptr_[_port_index], _host_ptr,
                                  _size_in_byte, cudaMemcpyHostToDevice));

      } else
        std::cerr << "device_memory_ports::send(const int&, value_type*)\t "
                     "device pointer " << _port_index
                  << " unitialized (nothing to transfer)\n";
    } else {
      std::cerr
          << "device_memory_ports::send(const int&, value_type*)\t port_index "
          << _port_index << " out of bounds (max: " << num_ports << ")\n";
    }
  }

  template <int from_port, int to_port>
  void sync() {
    BOOST_STATIC_ASSERT_MSG(from_port < num_ports && to_port < num_ports,
                            "device_memory_ports::sync<from,to>() \t trying to "
                            "setup memory region that is not under control");
    if (device_ptr_[from_port] && device_ptr_[to_port]) {
      HANDLE_ERROR(cudaStreamSynchronize(ports_[to_port]));
      HANDLE_ERROR(cudaStreamSynchronize(ports_[from_port]));
      HANDLE_ERROR(cudaMemcpy(device_ptr_[to_port], device_ptr_[from_port],
                              device_size_in_byte_[to_port],
                              cudaMemcpyHostToDevice));
    } else {
      std::cerr << "device_memory_ports::sync<from,to>()\t device pointer(s) "
                << from_port << " (" << device_ptr_[from_port] << ") "
                << to_port << " (" << device_ptr_[to_port]
                << ") unitialized (nothing to transfer)\n";
    }
  }

  template <int port_index>
  void receive(value_type* _host_ptr, unsigned long _size_in_byte = 0) {
    if (!_size_in_byte) _size_in_byte = device_size_in_byte_[port_index];

    if (_size_in_byte > device_size_in_byte_[port_index]) {
      std::cerr << "device_memory_ports::receive(const int&, value_type*)\t "
                   "size given exceeds allocated size (given: " << _size_in_byte
                << " B, allocated: " << device_size_in_byte_[port_index]
                << " B)\n";
      _size_in_byte = device_size_in_byte_[port_index];
    }

    BOOST_STATIC_ASSERT_MSG(port_index < num_ports,
                            "device_memory_ports::receive(value_type*) \t "
                            "trying to setup memory region that is not under "
                            "control");
    if (device_ptr_[port_index]) {
      if (ports_.count(port_index))
        HANDLE_ERROR(cudaMemcpyAsync(_host_ptr, device_ptr_[port_index],
                                     _size_in_byte, cudaMemcpyDeviceToHost,
                                     ports_[port_index]));
      else
        HANDLE_ERROR(cudaMemcpy(_host_ptr, device_ptr_[port_index],
                                _size_in_byte, cudaMemcpyDeviceToHost));

    } else
      std::cerr << "device_memory_ports::receive(value_type*)\t device pointer "
                << port_index << " unitialized (nothing to transfer)\n";
  }

  void receive(const int& _port_index, value_type* _host_ptr,
               unsigned long _size_in_byte = 0) {

    if (!_size_in_byte) _size_in_byte = device_size_in_byte_[_port_index];

    if (_size_in_byte > device_size_in_byte_[_port_index]) {
      std::cerr << "device_memory_ports::receive(const int&, value_type*, "
                   "unsigned long)\t size given exceeds allocated size (given: "
                << _size_in_byte
                << " B, allocated: " << device_size_in_byte_[_port_index]
                << " B)\n";
      _size_in_byte = device_size_in_byte_[_port_index];
    }

    if (_port_index < num_ports) {
      if (device_ptr_[_port_index]) {
        if (ports_.count(_port_index))
          HANDLE_ERROR(cudaMemcpyAsync(_host_ptr, device_ptr_[_port_index],
                                       _size_in_byte, cudaMemcpyDeviceToHost,
                                       ports_[_port_index]));
        else
          HANDLE_ERROR(cudaMemcpy(_host_ptr, device_ptr_[_port_index],
                                  _size_in_byte, cudaMemcpyDeviceToHost));

      } else
        std::cerr << "device_memory_ports::receive(const int&, value_type*)\t "
                     "device pointer " << _port_index
                  << " unitialized (nothing to transfer)\n";
    } else {
      std::cerr << "device_memory_ports::receive(const int&, value_type*)\t "
                   "port_index " << _port_index
                << " out of bounds (max: " << num_ports << ")\n";
    }
  }

  void sync_receive(const int& _port_index, value_type* _host_ptr,
                    unsigned long _size_in_byte = 0) {

    if (!_size_in_byte) _size_in_byte = device_size_in_byte_[_port_index];

    if (_size_in_byte > device_size_in_byte_[_port_index]) {
      std::cerr << "device_memory_ports::receive(const int&, value_type*, "
                   "unsigned long)\t size given exceeds allocated size (given: "
                << _size_in_byte
                << " B, allocated: " << device_size_in_byte_[_port_index]
                << " B)\n";
      _size_in_byte = device_size_in_byte_[_port_index];
    }

    if (_port_index < num_ports) {
      if (device_ptr_[_port_index]) {
        HANDLE_ERROR(cudaStreamSynchronize(ports_[_port_index]));
        HANDLE_ERROR(cudaMemcpy(_host_ptr, device_ptr_[_port_index],
                                _size_in_byte, cudaMemcpyDeviceToHost));

      } else
        std::cerr << "device_memory_ports::receive(const int&, value_type*)\t "
                     "device pointer " << _port_index
                  << " unitialized (nothing to transfer)\n";
    } else {
      std::cerr << "device_memory_ports::receive(const int&, value_type*)\t "
                   "port_index " << _port_index
                << " out of bounds (max: " << num_ports << ")\n";
    }
  }

  template <int port_index>
  cudaStream_t* stream() {
    BOOST_STATIC_ASSERT_MSG(
        port_index < num_ports,
        "device_memory_ports::stream<int>() \t invalid port number");
    if (ports_.count(port_index))
      return &ports_[port_index];
    else
      return 0;
  }

  template <int port_index>
  void sync_stream() {
    BOOST_STATIC_ASSERT_MSG(
        port_index < num_ports,
        "device_memory_ports::sync_stream<int>() \t invalid port number");
    if (ports_.count(port_index))
      HANDLE_ERROR(cudaStreamSynchronize(ports_[port_index]));
  }

  cudaStream_t* stream(const unsigned& _num) {

    if (_num < num_ports && ports_.count(_num))
      return &ports_[_num];
    else
      return 0;
  }

  template <int first_port, int second_port>
  void streams_of_two(std::vector<cudaStream_t*>& _cont) {
    BOOST_STATIC_ASSERT_MSG(first_port < num_ports && second_port && num_ports,
                            "device_memory_ports::streams_of_two() \t trying "
                            "to setup memory region that is not under control");

    if (_cont.size() >= 2) {
      _cont[0] = ports_.count(first_port) ? &ports_[first_port] : 0;
      _cont[1] = ports_.count(second_port) ? &ports_[second_port] : 0;
    } else {
      std::cerr << "device_memory_ports::streams_of_two() \t received vector "
                   "has incorrect size (received: " << _cont.size()
                << ", expected: >= 2\n";
    }
  }

  template <int num>
  value_type* at() {
    BOOST_STATIC_ASSERT_MSG(num < num_ports,
                            "device_memory_ports::at() \t trying to access "
                            "device pointer that does not exist");
    return device_ptr_[num];
  }

  value_type* at(const unsigned& _num) {
    if (_num < num_ports)
      return device_ptr_[_num];
    else
      return 0;
  }

  template <int num>
  value_type* sync_at() {
    BOOST_STATIC_ASSERT_MSG(num < num_ports,
                            "device_memory_ports::at() \t trying to access "
                            "device pointer that does not exist");
    HANDLE_ERROR(cudaStreamSynchronize(ports_[num]));
    return device_ptr_[num];
  }

  value_type* sync_at(const unsigned& _num) {
    if (_num < num_ports) {
      HANDLE_ERROR(cudaStreamSynchronize(ports_[_num]));
      return device_ptr_[_num];
    } else
      return 0;
  }
};

template <typename Tp>
class pinned_allocator {
 public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef Tp* pointer;
  typedef const Tp* const_pointer;
  typedef Tp& reference;
  typedef const Tp& const_reference;
  typedef Tp value_type;

  template <typename Tp1>
  struct rebind {
    typedef pinned_allocator<Tp1> other;
  };

  pinned_allocator() throw() {}
  pinned_allocator(const pinned_allocator&) throw() {}
  template <typename Tp1>
  pinned_allocator(const pinned_allocator<Tp1>&) throw() {}
  ~pinned_allocator() throw() {}

  pointer allocate(size_type n, const void* = 0) {
    pointer value = 0;
    HANDLE_ERROR(cudaMallocHost((void**)&value, n * sizeof(Tp)));
    // pointer value = static_cast<Tp*>(fftw_c_api<Tp>::malloc(n * sizeof(Tp)));
    return value;
  }

  void deallocate(pointer p, size_type) {
    HANDLE_ERROR(cudaFreeHost(p));
    // fftw_c_api<Tp>::free(p);
    return;
  }

  size_type max_size() const { return size_t(-1) / sizeof(Tp); }

  void construct(pointer p) { ::new ((void*)p) Tp(); }
  void construct(pointer p, const Tp& val) { ::new ((void*)p) Tp(val); }
  void destroy(pointer p) { p->~Tp(); }
};

};  // multiviewnative namespace

#endif /* _CUDA_MEMORY_H_ */

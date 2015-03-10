#ifndef _MANAGED_ALLOCATOR_H_
#define _MANAGED_ALLOCATOR_H_
#include "cuda_helpers.cuh"

template <typename Tp>
class managed_allocator {
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
    typedef managed_allocator<Tp1> other;
  };

  managed_allocator() throw() {}
  managed_allocator(const managed_allocator&) throw() {}
  template <typename Tp1>
  managed_allocator(const managed_allocator<Tp1>&) throw() {}
  ~managed_allocator() throw() {}

  pointer allocate(size_type n, const void* = 0) {
    Tp* data = 0;
    HANDLE_ERROR(cudaMallocManaged(&data, sizeof(Tp)*n));
    return data;
    // return static_cast<Tp*>(//managed_c_api<Tp>::malloc(n * sizeof(Tp))
    // 			    );
  }

  void deallocate(pointer p, size_type) { 
    HANDLE_ERROR(cudaFree(p));
    //  managed_c_api<Tp>::free(p); 
  }

  size_type max_size() const { return size_t(-1) / sizeof(Tp); }

  void construct(pointer p) { ::new ((void*)p) Tp(); }
  
  void construct(pointer p, const Tp& val) { ::new ((void*)p) Tp(val); }
  
  void destroy(pointer p) { p->~Tp(); }
};

#include "boost/multi_array.hpp"
typedef boost::multi_array<float, 3, managed_allocator<float> > managed_image_stack;

#endif /* _MANAGED_ALLOCATOR_H_ */

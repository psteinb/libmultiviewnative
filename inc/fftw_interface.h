#ifndef _FFTW_INTERFACE_H_
#define _FFTW_INTERFACE_H_

// seems that fftw coming with fedora 20 has this problem as documented here
// http://stackoverflow.com/questions/23165409/fftw-3-3-compile-error-using-nvcc-on-linux
// // if the following 3 lines are commented out

#ifndef __float128
#include <cfloat>
// check if the mantnisse of double representation is smaller,
// than long double (would perhaps not be the case on 32-bit
// systems, TODO!)
#if DBL_MANT_DIG < LDBL_MANT_DIG
#define __float128 long double
#endif

#endif

#include "fftw3.h"

// motivated by
// https://gitorious.org/cpp-bricks/fftw/source/023d40c478bcf45d7f7bea0c1e9a02ff68214875

template <typename T>
struct fftw_c_api {};

template <>
struct fftw_c_api<float> {

  static void* malloc(size_t n) { return fftwf_malloc(n); }

  static void free(void* p) { fftwf_free(p); }
};

template <>
struct fftw_c_api<double> {

  static void* malloc(size_t n) { return fftw_malloc(n); }

  static void free(void* p) { fftw_free(p); }
};

template <typename Tp>
class fftw_allocator {
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
    typedef fftw_allocator<Tp1> other;
  };

  fftw_allocator() throw() {}
  fftw_allocator(const fftw_allocator&) throw() {}
  template <typename Tp1>
  fftw_allocator(const fftw_allocator<Tp1>&) throw() {}
  ~fftw_allocator() throw() {}

  pointer allocate(size_type n, const void* = 0) {
    return static_cast<Tp*>(fftw_c_api<Tp>::malloc(n * sizeof(Tp)));
  }

  void deallocate(pointer p, size_type) { fftw_c_api<Tp>::free(p); }

  size_type max_size() const { return size_t(-1) / sizeof(Tp); }

  void construct(pointer p) { ::new ((void*)p) Tp(); }
  void construct(pointer p, const Tp& val) { ::new ((void*)p) Tp(val); }
  void destroy(pointer p) { p->~Tp(); }
};

template <typename PrecisionType>
struct fftw_api_definitions {
  // not defined and will throw a compiler error
};

template <>
struct fftw_api_definitions<float> {

  typedef fftwf_plan plan_type;
  typedef fftwf_complex complex_type;
  typedef float real_type;

  typedef void (*execute_plan_fptr)(const plan_type);
  static execute_plan_fptr execute_plan;

  typedef void (*reuse_plan_r2c_fptr)(const plan_type, real_type*,
                                      complex_type*);
  static reuse_plan_r2c_fptr reuse_plan_r2c;

  typedef void (*reuse_plan_c2r_fptr)(const plan_type, complex_type*,
                                      real_type*);
  static reuse_plan_c2r_fptr reuse_plan_c2r;

  typedef void (*destroy_plan_fptr)(plan_type);
  static destroy_plan_fptr destroy_plan;

  typedef plan_type (*dft_r2c_3d_fptr)(int, int, int, real_type*, complex_type*,
                                       unsigned);

  typedef plan_type (*dft_c2r_3d_fptr)(int, int, int, complex_type*, float*,
                                       unsigned);

  static dft_c2r_3d_fptr dft_c2r_3d;
  static dft_r2c_3d_fptr dft_r2c_3d;

  // fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany,
  //                                double *in, const int *inembed,
  //                                int istride, int idist,
  //                                fftw_complex *out, const int *onembed,
  //                                int ostride, int odist,
  //                                unsigned flags);
  typedef plan_type (*dft_r2c_many_fptr)(int, const int*, int, real_type*,
                                         const int*, int, int, complex_type*,
                                         const int*, int, int, unsigned);
  // fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany,
  //                                      fftw_complex *in, const int *inembed,
  //                                      int istride, int idist,
  //                                      double *out, const int *onembed,
  //                                      int ostride, int odist,
  //                                      unsigned flags);
  typedef plan_type (*dft_c2r_many_fptr)(int, const int*, int, complex_type*,
                                         const int*, int, int, real_type*,
                                         const int*, int, int, unsigned);

  static dft_c2r_many_fptr dft_c2r_many;
  static dft_r2c_many_fptr dft_r2c_many;

  typedef int (*init_threads_fptr)(void);
  static init_threads_fptr init_threads;

  typedef void (*plan_with_threads_fptr)(int);
  static plan_with_threads_fptr plan_with_threads;

  typedef void (*cleanup_threads_fptr)(void);
  static cleanup_threads_fptr cleanup_threads;

  // int fftw_init_threads(void);
  // void fftw_plan_with_nthreads(int nthreads);
  // void fftw_cleanup_threads(void);
};

fftw_api_definitions<float>::execute_plan_fptr
    fftw_api_definitions<float>::execute_plan = fftwf_execute;
fftw_api_definitions<float>::reuse_plan_r2c_fptr
    fftw_api_definitions<float>::reuse_plan_r2c = fftwf_execute_dft_r2c;
fftw_api_definitions<float>::reuse_plan_c2r_fptr
    fftw_api_definitions<float>::reuse_plan_c2r = fftwf_execute_dft_c2r;
fftw_api_definitions<float>::destroy_plan_fptr
    fftw_api_definitions<float>::destroy_plan = fftwf_destroy_plan;
fftw_api_definitions<float>::dft_c2r_3d_fptr
    fftw_api_definitions<float>::dft_c2r_3d = fftwf_plan_dft_c2r_3d;
fftw_api_definitions<float>::dft_r2c_3d_fptr
    fftw_api_definitions<float>::dft_r2c_3d = fftwf_plan_dft_r2c_3d;
fftw_api_definitions<float>::dft_c2r_many_fptr
    fftw_api_definitions<float>::dft_c2r_many = fftwf_plan_many_dft_c2r;
fftw_api_definitions<float>::dft_r2c_many_fptr
    fftw_api_definitions<float>::dft_r2c_many = fftwf_plan_many_dft_r2c;

fftw_api_definitions<float>::init_threads_fptr
    fftw_api_definitions<float>::init_threads = fftwf_init_threads;
fftw_api_definitions<float>::plan_with_threads_fptr
    fftw_api_definitions<float>::plan_with_threads = fftwf_plan_with_nthreads;
fftw_api_definitions<float>::cleanup_threads_fptr
    fftw_api_definitions<float>::cleanup_threads = fftwf_cleanup_threads;

template <>
struct fftw_api_definitions<double> {

  typedef fftw_plan plan_type;
  typedef fftw_complex complex_type;
  typedef double real_type;

  typedef void (*execute_plan_fptr)(const plan_type);
  static execute_plan_fptr execute_plan;

  typedef void (*reuse_plan_r2c_fptr)(const plan_type, real_type*,
                                      complex_type*);
  static reuse_plan_r2c_fptr reuse_plan_r2c;

  typedef void (*reuse_plan_c2r_fptr)(const plan_type, complex_type*,
                                      real_type*);
  static reuse_plan_c2r_fptr reuse_plan_c2r;

  typedef void (*destroy_plan_fptr)(plan_type);
  static destroy_plan_fptr destroy_plan;

  typedef plan_type (*dft_r2c_3d_fptr)(int, int, int, real_type*, complex_type*,
                                       unsigned);
  typedef plan_type (*dft_c2r_3d_fptr)(int, int, int, complex_type*, double*,
                                       unsigned);

  static dft_c2r_3d_fptr dft_c2r_3d;
  static dft_r2c_3d_fptr dft_r2c_3d;

  // fftw_plan fftw_plan_many_dft_r2c(int rank, const int *n, int howmany,
  //                                double *in, const int *inembed,
  //                                int istride, int idist,
  //                                fftw_complex *out, const int *onembed,
  //                                int ostride, int odist,
  //                                unsigned flags);
  typedef plan_type (*dft_r2c_many_fptr)(int, const int*, int, real_type*,
                                         const int*, int, int, complex_type*,
                                         const int*, int, int, unsigned);
  // fftw_plan fftw_plan_many_dft_c2r(int rank, const int *n, int howmany,
  //                                      fftw_complex *in, const int *inembed,
  //                                      int istride, int idist,
  //                                      double *out, const int *onembed,
  //                                      int ostride, int odist,
  //                                      unsigned flags);
  typedef plan_type (*dft_c2r_many_fptr)(int, const int*, int, complex_type*,
                                         const int*, int, int, real_type*,
                                         const int*, int, int, unsigned);

  static dft_c2r_many_fptr dft_c2r_many;
  static dft_r2c_many_fptr dft_r2c_many;

  typedef int (*init_threads_fptr)(void);
  static init_threads_fptr init_threads;

  typedef void (*plan_with_threads_fptr)(int);
  static plan_with_threads_fptr plan_with_threads;

  typedef void (*cleanup_threads_fptr)(void);
  static cleanup_threads_fptr cleanup_threads;
};

fftw_api_definitions<double>::execute_plan_fptr
    fftw_api_definitions<double>::execute_plan = fftw_execute;
fftw_api_definitions<double>::reuse_plan_r2c_fptr
    fftw_api_definitions<double>::reuse_plan_r2c = fftw_execute_dft_r2c;
fftw_api_definitions<double>::reuse_plan_c2r_fptr
    fftw_api_definitions<double>::reuse_plan_c2r = fftw_execute_dft_c2r;
fftw_api_definitions<double>::destroy_plan_fptr
    fftw_api_definitions<double>::destroy_plan = fftw_destroy_plan;
fftw_api_definitions<double>::dft_c2r_3d_fptr
    fftw_api_definitions<double>::dft_c2r_3d = fftw_plan_dft_c2r_3d;
fftw_api_definitions<double>::dft_r2c_3d_fptr
    fftw_api_definitions<double>::dft_r2c_3d = fftw_plan_dft_r2c_3d;
fftw_api_definitions<double>::dft_c2r_many_fptr
    fftw_api_definitions<double>::dft_c2r_many = fftw_plan_many_dft_c2r;
fftw_api_definitions<double>::dft_r2c_many_fptr
    fftw_api_definitions<double>::dft_r2c_many = fftw_plan_many_dft_r2c;

fftw_api_definitions<double>::init_threads_fptr
    fftw_api_definitions<double>::init_threads = fftw_init_threads;
fftw_api_definitions<double>::plan_with_threads_fptr
    fftw_api_definitions<double>::plan_with_threads = fftw_plan_with_nthreads;
fftw_api_definitions<double>::cleanup_threads_fptr
    fftw_api_definitions<double>::cleanup_threads = fftw_cleanup_threads;

#endif /* _FFTW_INTERFACE_H_ */

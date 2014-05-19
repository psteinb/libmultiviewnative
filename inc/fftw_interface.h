#ifndef _FFTW_INTERFACE_H_
#define _FFTW_INTERFACE_H_

#include "fftw3.h"


//motivated by https://gitorious.org/cpp-bricks/fftw/source/023d40c478bcf45d7f7bea0c1e9a02ff68214875

template <typename T>
struct fftw_c_api{
};

template <>
struct fftw_c_api<float>{

  static void *malloc(size_t n)
   { return fftwf_malloc(n); }

  static void free(void *p)
   { fftwf_free(p); }

};

template <>
struct fftw_c_api<double>{

  static void *malloc(size_t n)
   { return fftw_malloc(n); }

  static void free(void *p)
   { fftw_free(p); }

};


template <typename Tp>
class fftw_allocator
{
public:
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;
  typedef Tp*        pointer;
  typedef const Tp*  const_pointer;
  typedef Tp&        reference;
  typedef const Tp&  const_reference;
  typedef Tp         value_type;
 
  template<typename Tp1> struct rebind { typedef fftw_allocator<Tp1> other; };
 
  fftw_allocator() throw() { }
  fftw_allocator(const fftw_allocator&) throw() { }
  template<typename Tp1> fftw_allocator(const fftw_allocator<Tp1>&) throw() { }
  ~fftw_allocator() throw() { }
 
  pointer allocate(size_type n, const void* = 0)
  { return static_cast<Tp*>(fftw_c_api<Tp>::malloc(n * sizeof(Tp))); }
 
  void deallocate(pointer p, size_type) { fftw_c_api<Tp>::free(p); }
 
  size_type max_size() const { return size_t(-1) / sizeof(Tp); }
 
  void construct(pointer p) { ::new((void *)p) Tp(); }
  void construct(pointer p, const Tp& val) { ::new((void *)p) Tp(val); }
  void destroy(pointer p) { p->~Tp(); }
};

template <typename PrecisionType>
struct fftw_api_definitions {
  //not defined and will throw a compiler error
};

template <>
struct fftw_api_definitions<float> {
      
  typedef fftwf_plan plan_type;
  typedef fftwf_complex complex_type;
  typedef float real_type;

  typedef void (*execute_plan_ftr)(const plan_type);
  static execute_plan_ftr execute_plan;

  typedef void (*destroy_plan_ftr)(plan_type);
  static destroy_plan_ftr destroy_plan;

  typedef plan_type (*dft_r2c_3d_ftr)(int , int , int ,
				      float*, complex_type*,
				      unsigned);
  typedef plan_type (*dft_c2r_3d_ftr)(int , int , int ,
				      complex_type*,float*, 
				      unsigned);

  static  dft_c2r_3d_ftr dft_c2r_3d;
  static  dft_r2c_3d_ftr dft_r2c_3d;

};

fftw_api_definitions<float>::execute_plan_ftr  fftw_api_definitions<float>::execute_plan  =  fftwf_execute;
fftw_api_definitions<float>::destroy_plan_ftr  fftw_api_definitions<float>::destroy_plan  =  fftwf_destroy_plan;
fftw_api_definitions<float>::dft_c2r_3d_ftr    fftw_api_definitions<float>::dft_c2r_3d    =  fftwf_plan_dft_c2r_3d;
fftw_api_definitions<float>::dft_r2c_3d_ftr    fftw_api_definitions<float>::dft_r2c_3d    =  fftwf_plan_dft_r2c_3d;

template <>
struct fftw_api_definitions<double> {
      
  typedef fftw_plan plan_type;
  typedef fftw_complex complex_type;
  typedef double real_type;

  typedef void (*execute_plan_ftr)(const plan_type);
  static execute_plan_ftr execute_plan;

  typedef void (*destroy_plan_ftr)(plan_type);
  static destroy_plan_ftr destroy_plan;

  typedef plan_type (*dft_r2c_3d_ftr)(int , int , int ,
				      double*, complex_type*,
				      unsigned);
  typedef plan_type (*dft_c2r_3d_ftr)(int , int , int ,
				      complex_type*,double*, 
				      unsigned);

  static  dft_c2r_3d_ftr dft_c2r_3d;
  static  dft_r2c_3d_ftr dft_r2c_3d;

};

fftw_api_definitions<double>::execute_plan_ftr  fftw_api_definitions<double>::execute_plan  =  fftw_execute;
fftw_api_definitions<double>::destroy_plan_ftr  fftw_api_definitions<double>::destroy_plan  =  fftw_destroy_plan;
fftw_api_definitions<double>::dft_c2r_3d_ftr    fftw_api_definitions<double>::dft_c2r_3d    =  fftw_plan_dft_c2r_3d;
fftw_api_definitions<double>::dft_r2c_3d_ftr    fftw_api_definitions<double>::dft_r2c_3d    =  fftw_plan_dft_r2c_3d;



#endif /* _FFTW_INTERFACE_H_ */

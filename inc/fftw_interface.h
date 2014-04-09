#ifndef _FFTW_INTERFACE_H_
#define _FFTW_INTERFACE_H_

#include "fftw3.h"

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

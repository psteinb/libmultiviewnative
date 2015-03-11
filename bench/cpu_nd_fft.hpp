#ifndef _CPU_ND_FFT_H_
#define _CPU_ND_FFT_H_
#include "image_stack_utils.h"
#include "fftw_interface.h"

typedef fftw_api_definitions<float> fftw_api;
/**
   \brief calculates the expected memory consumption for an inplace r-2-c
   transform according to
   http://docs.nvidia.com/cuda/cufft/index.html#data-layout

   \param[in] _shape std::vector that contains the dimensions

   \return numbe of bytes consumed
   \retval

*/
unsigned long fftw_r2c_memory(const std::vector<unsigned>& _shape) {

  std::vector<unsigned> adapted(_shape);
  static std::vector<int> c_ordering(3);
  c_ordering[0] = 2;
  c_ordering[1] = 1;
  c_ordering[2] = 0;

  multiviewnative::adapt_shape_for_fftw_inplace(c_ordering, _shape, adapted );

  unsigned long value = std::accumulate(adapted.begin(), adapted.end(), 1u,
                                        std::multiplies<unsigned long>());
  value *= sizeof(fftw_api::complex_type);
  return value;
}

void fftw_r2c_reshape(std::vector<unsigned>& _shape) {

  static multiviewnative::image_stack dummy;
  std::vector<unsigned> adapted(_shape);
  static std::vector<int> c_ordering(3);
  c_ordering[0] = dummy.storage_order().ordering(0);
  c_ordering[1] = dummy.storage_order().ordering(1);
  c_ordering[2] = dummy.storage_order().ordering(2);
    

  multiviewnative::adapt_shape_for_fftw_inplace(c_ordering, _shape, adapted );

  std::copy(adapted.begin(), adapted.end(), _shape.begin());
}

/**
   \brief function that computes a r-2-c float32 FFT single-threaded
   the function assumes that there has been space pre-allocated on device and
   that the data required has already been transferred

   \param[in] _d_src_buffer was already allocated to match the expected size of
   the FFT
   \param[in] _d_dest_buffer if non-zero, was already allocated to match the
   expected size of the FFT and if non-zero will be used as destionation buffer
   \param[in] _stack host-side nD image

   \return
   \retval

*/
void st_fftw(const std::vector<unsigned>& _shape, float* _d_src_buffer,
             float* _d_dest_buffer = 0, fftw_api::plan_type* _plan = 0) {

  fftw_api::real_type* fourier_input = (fftw_api::real_type*)_d_src_buffer;
  fftw_api::complex_type* fourier_output =
      (fftw_api::complex_type*)_d_dest_buffer;
  if (!fourier_output) fourier_output = (fftw_api::complex_type*)fourier_input;

  if (!_plan) {
    _plan = new fftw_api::plan_type;

    *_plan = fftw_api::dft_r2c_3d(_shape[0], _shape[1], _shape[2],
                                  fourier_input, fourier_output, FFTW_ESTIMATE);
  }

  fftw_api::execute_plan(*_plan);

  if (!_plan) {
    fftw_api::destroy_plan(*_plan);
    delete _plan;
  }
}

/**
   \brief function that computes a r-2-c float32 FFT single-threaded
   the function assumes that there has been space pre-allocated on device and
   that the data required has already been transferred

   \param[in] _d_src_buffer was already allocated to match the expected size of
   the FFT
   \param[in] _d_dest_buffer if non-zero, was already allocated to match the
   expected size of the FFT and if non-zero will be used as destionation buffer
   \param[in] _stack host-side nD image

   \return
   \retval

*/
void reuse_fftw_plan(const std::vector<unsigned>& _shape, float* _d_src_buffer,
                     float* _d_dest_buffer = 0,
                     fftw_api::plan_type* _plan = 0) {

  fftw_api::real_type* fourier_input = (fftw_api::real_type*)_d_src_buffer;
  fftw_api::complex_type* fourier_output =
      (fftw_api::complex_type*)_d_dest_buffer;
  if (!fourier_output) fourier_output = (fftw_api::complex_type*)fourier_input;

  fftw_api::reuse_plan_r2c(*_plan, fourier_input, fourier_output);
}

#endif /* _CPU_ND_FFT_H_ */

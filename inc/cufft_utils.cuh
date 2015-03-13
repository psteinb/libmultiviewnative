#ifndef _CUFFT_UTILS_H_
#define _CUFFT_UTILS_H_
#include <vector>
#include <algorithm>

#include "cuda_helpers.cuh"
#include "cufft.h"
#include "cufft_interface.cuh"
#include "plan_store.cuh"

static void HandleCufftError(cufftResult_t err, const char* file, int line) {
  if (err != CUFFT_SUCCESS) {
    std::cerr << "cufftResult [" << err << "] in " << file << " at line "
              << line << std::endl;
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_CUFFT_ERROR(err) (HandleCufftError(err, __FILE__, __LINE__))

namespace multiviewnative {


template <typename TransferT,
          cufftCompatibility Mode = CUFFT_COMPATIBILITY_NATIVE>
class inplace_3d_transform_on_device {



 public:
  typedef gpu::cufft_api<TransferT> api; 
  typedef typename api::real_t real_type;
  typedef typename api::complex_t complex_type;
  typedef typename api::plan_t plan_type;

  typedef long size_type;

  static const int dimensionality = 3;

  template <typename DimT>
  inplace_3d_transform_on_device(TransferT* _input, DimT* _shape)
      : input_(_input), shape_(_shape, _shape + dimensionality) {
  }

  void forward(cudaStream_t* _stream = 0) {

    
    if(!gpu::plan_store<real_type>::get()->has_key(shape_))
      gpu::plan_store<real_type>::get()->add(shape_);
    
    plan_type* plan = gpu::plan_store<real_type>::get()->get_forward(shape_);

    if (_stream) 
      HANDLE_CUFFT_ERROR(cufftSetStream(*plan, *_stream));

    HANDLE_CUFFT_ERROR(
        cufftExecR2C(*plan, input_, (complex_type*)input_));


  };

  void backward(cudaStream_t* _stream = 0) {

    if(!gpu::plan_store<real_type>::get()->has_key(shape_))
      gpu::plan_store<real_type>::get()->add(shape_);
    
    plan_type* plan = gpu::plan_store<real_type>::get()->get_backward(shape_);

    if (_stream) 
      HANDLE_CUFFT_ERROR(cufftSetStream(*plan, *_stream));

    HANDLE_CUFFT_ERROR(
        cufftExecC2R(*plan, (complex_type*)input_, input_));

  };

  ~inplace_3d_transform_on_device() {};

 private:
  TransferT* input_;
  multiviewnative::shape_t shape_;


};
}
#endif /* _FFT_UTILS_H_ */

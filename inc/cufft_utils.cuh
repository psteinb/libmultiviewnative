#ifndef _CUFFT_UTILS_H_
#define _CUFFT_UTILS_H_
#include <vector>
#include <algorithm>

#include "cufft.h"

#include "point.h"
#include "cuda_helpers.cuh"

#include "cufft_interface.cuh"
#include "plan_store.cuh"

#define HANDLE_CUFFT_ERROR(err) (multiviewnative::gpu::HandleCufftError(err, __FILE__, __LINE__))

namespace multiviewnative {


template <typename TransferT,
          cufftCompatibility Mode = CUFFT_COMPATIBILITY_NATIVE>
class inplace_3d_transform_on_device {



 public:
  typedef gpu::cufft_api<TransferT> api; 
  typedef typename api::real_t real_type;
  typedef typename api::complex_t complex_type;
  typedef typename api::plan_t plan_type;
  typedef gpu::plan_store<real_type> plan_store;

  typedef long size_type;

  static const int dimensionality = 3;

  template <typename DimT>
  inplace_3d_transform_on_device(TransferT* _input, DimT* _shape)
      : input_(_input), shape_(_shape, _shape + dimensionality) {
  }

  void forward(cudaStream_t* _stream = 0) {

    
    if(!plan_store::get()->has_key(shape_))
      plan_store::get()->add(shape_);
    
    plan_type* plan = plan_store::get()->get_forward(shape_);

    if (_stream) 
      HANDLE_CUFFT_ERROR(cufftSetStream(*plan, *_stream));

    HANDLE_CUFFT_ERROR(
        cufftExecR2C(*plan, input_, (complex_type*)input_));


  };

  void backward(cudaStream_t* _stream = 0) {

    if(!plan_store::get()->has_key(shape_))
      plan_store::get()->add(shape_);
    
    plan_type* plan = plan_store::get()->get_backward(shape_);

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

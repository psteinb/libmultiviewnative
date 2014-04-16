#ifndef _CUFFT_UTILS_H_
#define _CUFFT_UTILS_H_
#include <vector>
#include <algorithm>

#include "cuda_helpers.cuh"
#include "cufft.h"

namespace multiviewnative {


static void HandleCufftError( cufftResult_t err,
                         const char *file,
                         int line ) {
    if (err != CUFFT_SUCCESS) {
      std::cerr << "cufftResult [" << err << "] in " << file << " at line " << line << std::endl;
      cudaDeviceReset();
      exit( EXIT_FAILURE );
    }
}

#define HANDLE_CUFFT_ERROR( err ) (HandleCufftError( err, __FILE__, __LINE__ ))

  template <typename TransferT, cufftCompatibility Mode = CUFFT_COMPATIBILITY_NATIVE>
class inplace_3d_transform_on_device {
  
  static const int dimensionality = 3;

public:
  
  typedef  cufftReal     real_type;
  typedef  long          size_type;
  typedef  cufftComplex  complex_type;
  typedef  cufftHandle   plan_type;

  template <typename DimT>
  inplace_3d_transform_on_device(TransferT* _input, DimT* _shape):
    input_(_input),
    shape_(dimensionality,0),
    transform_plan_()
  {
    std::copy(_shape, _shape + dimensionality, shape_.begin());
    
  }

  
  void forward(){
    
    HANDLE_CUFFT_ERROR(cufftPlan3d(&transform_plan_, (int)shape_[0], (int)shape_[1], (int)shape_[2], CUFFT_R2C));
    HANDLE_CUFFT_ERROR(cufftSetCompatibilityMode(transform_plan_,Mode));
    HANDLE_CUFFT_ERROR(cufftExecR2C(transform_plan_, input_, (cufftComplex *)input_));

    clean_up();
  };


  void backward(){
    
    HANDLE_CUFFT_ERROR(cufftPlan3d(&transform_plan_, (int)shape_[0], (int)shape_[1], (int)shape_[2], CUFFT_C2R));
    HANDLE_CUFFT_ERROR(cufftSetCompatibilityMode(transform_plan_,Mode));
    HANDLE_CUFFT_ERROR(cufftExecC2R(transform_plan_,(cufftComplex *)input_, input_));

    clean_up();
  };
  
  

  ~inplace_3d_transform_on_device(){
  };

private:
      
  TransferT* input_;
  std::vector<size_type> shape_; 
  plan_type transform_plan_;

  void clean_up(){
    HANDLE_CUFFT_ERROR( cufftDestroy(transform_plan_) );
  };

};


}
#endif /* _FFT_UTILS_H_ */

#ifndef _CUFFT_INTERFACE_H_
#define _CUFFT_INTERFACE_H_

#include "cufft.h"

namespace multiviewnative {

  namespace mvn = multiviewnative;

  

  namespace gpu {

    template <typename T>
    struct cufft_api{};

    template <>
    struct cufft_api<float>{

      typedef cufftReal real_t;
      typedef cufftComplex complex_t;
      typedef cufftHandle plan_t;

    };

    template <>
    struct cufft_api<double>{

      typedef cufftDoubleReal real_t;
      typedef cufftDoubleComplex complex_t;
      typedef cufftHandle plan_t;

    };

  };

};

#endif /* _CUFFT_INTERFACE_H_ */

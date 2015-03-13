#ifndef _CUFFT_INTERFACE_H_
#define _CUFFT_INTERFACE_H_

#include "cufft.h"

namespace multiviewnative {


  namespace gpu {


    static void HandleCufftError(cufftResult_t err, const char* file, int line) {
      if (err != CUFFT_SUCCESS) {
	std::cerr << "cufftResult [" << err << "] in " << file << " at line "
		  << line << std::endl;
	cudaDeviceReset();
	exit(EXIT_FAILURE);
      }
    }


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

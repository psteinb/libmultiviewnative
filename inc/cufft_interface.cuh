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


    /**
       \brief calculates the expected memory consumption for an inplace r-2-c
       transform according to
       http://docs.nvidia.com/cuda/cufft/index.html#data-layout

       \param[in] _shape_begin begin iterator of shape array
       \param[in] _shape_end end iterator of shape array

       \return numbe of bytes consumed
       \retval

    */
    template <typename Iter>
    unsigned long cufft_r2c_memory(Iter _shape_begin, Iter _shape_end) {

      unsigned long start = 1;
      unsigned long value = std::accumulate(_shape_begin, _shape_end - 1, start,
					    std::multiplies<unsigned long>());
      value *= (std::floor(*(_shape_end - 1) / 2.) + 1) * sizeof(cufftComplex);
      return value;
    }

    template <typename Iter>
    std::vector<int> cufft_r2c_shape(Iter _shape_begin, Iter _shape_end) {

      std::vector<int> value(_shape_begin, _shape_end);
      value[value.size()-1] = 2*(std::floor(value.back() / 2.) + 1);
      return value;
  
    }

    /**
       \brief calculates the expected memory consumption for an inplace r-2-c
       transform according to
       http://docs.nvidia.com/cuda/cufft/index.html#data-layout

       \param[in] _shape std::vector that contains the dimensions

       \return numbe of bytes consumed
       \retval

    */
    unsigned long cufft_r2c_memory(const std::vector<unsigned>& _shape) {

      unsigned long start = 1;
      unsigned long value = std::accumulate(_shape.begin(), _shape.end() - 1, start,
					    std::multiplies<unsigned long>());
      value *= (std::floor(_shape.back() / 2.) + 1) * sizeof(cufftComplex);
      return value;
    }

    template <typename T>
    std::vector<T> cufft_r2c_shape(const std::vector<T>& _shape){
  
      std::vector<T> value(_shape.begin(), _shape.end());
      value[value.size()-1] = 2*(std::floor(_shape.back() / 2.) + 1);
      return value;
  
    }

  };

};

#endif /* _CUFFT_INTERFACE_H_ */

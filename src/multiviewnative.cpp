#define __MULTIVIEWNATIVE_CPP__

#include <vector>
#include <cmath>

#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"

#include "cpu_kernels.h"

namespace mvn = multiviewnative;


//typedef fftw_api_definitions<float> fftwf_api;

template <typename T>
bool check_nan(T* _array, const size_t& _size) {

  bool value = false;
  for (size_t i = 0; i < _size; ++i) {
    if (std::isnan(_array[i])) {
      value = true;
      break;
    }
  }

  return value;
}

template <typename T>
bool check_inf(T* _array, const size_t& _size) {

  bool value = false;
  for (size_t i = 0; i < _size; ++i) {
    if (std::isinf(_array[i])) {
      value = true;
      break;
    }
  }

  return value;
}

template <typename T>
bool check_malformed_float(T* _array, const size_t& _size) {

  bool value = false;
  for (size_t i = 0; i < _size; ++i) {
    if (std::isinf(_array[i]) || std::isnan(_array[i])) {
      value = true;
      break;
    }
  }

  return value;
}



// implements http://arxiv.org/abs/1308.0730 (Eq 70)
namespace multiviewnative {

  namespace cpu {
    
    template <typename Tag>
    struct convolve{};

    template <>
    struct convolve<serial_tag>{
      
      typedef mvn::cpu_convolve<> type;
      typedef type::transform_policy transform_type;
      typedef type::padding_policy padding_type;
      
    };

    template <>
    struct convolve<parallel_tag>{
      
      typedef mvn::cpu_convolve<mvn::parallel_inplace_3d_transform> type;
      typedef type::transform_policy transform_type;
      typedef type::padding_policy padding_type;
      
    };    
    
    template <typename Tag>
    void inplace_cpu_deconvolve(imageType* psi, 
				workspace input, 
				double lambda,
                                imageType minValue,
				int nthreads = 1
				) {
      
      typedef typename convolve<Tag>::transform_type transform_t;
      typedef typename convolve<Tag>::padding_type padding_t;
      typedef typename convolve<Tag>::type fold_t;
      
      // lay the kernel pointers aside
      std::vector<mvn::image_stack_ref> kernel1_ptr;
      std::vector<mvn::image_stack_ref> kernel2_ptr;
      std::vector<mvn::shape_t> image_shapes(input.num_views_);
      std::vector<mvn::shape_t> kernel1_shapes(input.num_views_);
      std::vector<mvn::shape_t> kernel2_shapes(input.num_views_);

      for (int v = 0; v < input.num_views_; ++v) {
        kernel1_shapes[v] =
            mvn::shape_t(input.data_[v].kernel1_dims_,
                         input.data_[v].kernel1_dims_ +
                             mvn::image_stack_ref::dimensionality);
        kernel1_ptr.push_back(
            mvn::image_stack_ref(input.data_[v].kernel1_, kernel1_shapes[v]));

        kernel2_shapes[v] =
            mvn::shape_t(input.data_[v].kernel2_dims_,
                         input.data_[v].kernel2_dims_ +
                             mvn::image_stack_ref::dimensionality);
        kernel2_ptr.push_back(
            mvn::image_stack_ref(input.data_[v].kernel2_, kernel2_shapes[v]));
      }

      // create the kernels in memory (this will double the memory consumption)
      std::vector<mvn::fftw_image_stack> forwarded_kernel1(input.num_views_);
      std::vector<mvn::fftw_image_stack> forwarded_kernel2(input.num_views_);

      for (int v = 0; v < input.num_views_; ++v) {

        image_shapes[v] = mvn::shape_t(
            input.data_[v].image_dims_,
            input.data_[v].image_dims_ + mvn::image_stack_ref::dimensionality);

        transform_t fft(image_shapes[v]);

        padding_t k1_padder(&(image_shapes[v])[0],
			  &(kernel1_shapes[v])[0]);
        padding_t k2_padder(&(image_shapes[v])[0],
			  &(kernel2_shapes[v])[0]);

        // prepare the kernels for fft forward transform
        forwarded_kernel1[v].resize(image_shapes[v]);
        k1_padder.wrapped_insert_at_offsets(kernel1_ptr[v],
                                            forwarded_kernel1[v]);
        fft.padd_for_fft(&forwarded_kernel1[v]);

        forwarded_kernel2[v].resize(image_shapes[v]);
        k2_padder.wrapped_insert_at_offsets(kernel2_ptr[v],
                                            forwarded_kernel2[v]);
        fft.padd_for_fft(&forwarded_kernel2[v]);

        // call fft
        fft.forward(&forwarded_kernel1[v]);
        fft.forward(&forwarded_kernel2[v]);
      }

      // do the convolution
      mvn::image_stack_ref input_psi(psi, image_shapes[0]);
      mvn::image_stack integral = input_psi;

      view_data view_access;

      for (int it = 0; it < input.num_iterations_; ++it) {
        for (unsigned view = 0; view < input.num_views_; ++view) {

          view_access = input.data_[view];
          integral = input_psi;

          // convolve: psi x kernel1 -> psiBlurred :: (Psi*P_v)
          fold_t convolver1(integral.data(),
                                         &(image_shapes[view])[0],
                                         view_access.kernel1_dims_);
          convolver1.half_inplace(forwarded_kernel1[view]);

          // view / psiBlurred -> psiBlurred :: (phi_v / (Psi*P_v))
          compute_quotient<Tag>(view_access.image_, 
			   integral.data(),
			   input_psi.num_elements(),
			   nthreads);

          // convolve: psiBlurred x kernel2 -> integral :: (phi_v / (Psi*P_v)) *
          // P_v^{compound}
          fold_t convolver2(integral.data(),
				   &(image_shapes[view])[0],
				   view_access.kernel2_dims_);
          convolver2.half_inplace(forwarded_kernel2[view]);

          // computeFinalValues(input_psi,integral,weights)
          // studied impact of different techniques on how to implement this
          // decision (decision in object, decision in if clause)
          // compiler opt & branch prediction seems to suggest this solution
          if (lambda > 0)
            regularized_final_values<Tag>(input_psi.data(), 
					  integral.data(), 
					  view_access.weights_,
					  input_psi.num_elements(), 
					  lambda, 
					  minValue,
					  0,//for array offset
					  nthreads);
          else
            final_values<Tag>(input_psi.data(), 
			      integral.data(),
			      view_access.weights_, 
			      input_psi.num_elements(),
			      minValue,
			      0,//for array offset
			      nthreads);
        }
      }

      // put kernel pointers back to keep memory clean
      for (int v = 0; v < input.num_views_; ++v) {
        input.data_[v].kernel1_ = kernel1_ptr[v].data();
        input.data_[v].kernel2_ = kernel2_ptr[v].data();
      }
    }
  }
}

void inplace_cpu_deconvolve(imageType* psi, workspace input, int nthreads) {

  // launch deconvolution
  if (nthreads == 1)
    mvn::cpu::inplace_cpu_deconvolve<mvn::cpu::serial_tag>(psi, input, input.lambda_, input.minValue_);
  else{
    mvn::cpu::convolve<mvn::cpu::parallel_tag>::transform_type::set_n_threads(nthreads);
    mvn::cpu::inplace_cpu_deconvolve<mvn::cpu::parallel_tag>(psi, input, 
						   input.lambda_,
						   input.minValue_,
						   nthreads);
  }
}

/**
   \brief inplace cpu based convolution, decides upon input value of nthreads
   whether to use single-threaded or multi-threaded implementation

   \param[in] im 1D array that contains the data image stack
   \param[in] imDim 3D array that contains the shape of the image stack im
   \param[in] kernel 1D array that contains the data kernel stack
   \param[in] kernelDim 3D array that contains the shape of the kernel stack
   kernel
   \param[in] nthreads number of threads to use

   \return
   \retval

*/
void inplace_cpu_convolution(imageType* im, 
			     int* imDim, 
			     imageType* kernel,
                             int* kernelDim, 
			     int nthreads) {

  unsigned image_dim[3];
  unsigned kernel_dim[3];

  std::copy(imDim, imDim + 3, &image_dim[0]);
  std::copy(kernelDim, kernelDim + 3, &kernel_dim[0]);

  if (nthreads != 1) {
    mvn::cpu::convolve<mvn::cpu::parallel_tag>::type convolver(im, image_dim, kernel, kernel_dim);
    mvn::cpu::convolve<mvn::cpu::parallel_tag>::transform_type::set_n_threads(nthreads);
    convolver.inplace();
  } else {
    mvn::cpu::convolve<mvn::cpu::serial_tag>::type convolver(im, image_dim, kernel, kernel_dim);
    convolver.inplace();
  }
}

#define __MULTIVIEWNATIVE_CPP__

#include <vector>
#include <cmath>

#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"

#include "cpu_kernels.h"

typedef multiviewnative::zero_padd<multiviewnative::image_stack> wrap_around_padding;
typedef multiviewnative::inplace_3d_transform<multiviewnative::image_stack> serial_transform;
typedef multiviewnative::parallel_inplace_3d_transform<multiviewnative::image_stack> parallel_transform;
typedef multiviewnative::cpu_convolve<wrap_around_padding,imageType, unsigned> default_convolution;

template <typename T>
bool check_nan(T* _array, const size_t& _size){

  bool value = false;
  for(size_t i = 0;i<_size;++i){
    if(std::isnan(_array[i])){
	value = true;
	break;
    }
       
  }

  return value;
}

template <typename T>
bool check_inf(T* _array, const size_t& _size){

  bool value = false;
  for(size_t i = 0;i<_size;++i){
    if(std::isinf(_array[i])){
	value = true;
	break;
    }
       
  }

  return value;
}

template <typename T>
bool check_malformed_float(T* _array, const size_t& _size){

  bool value = false;
  for(size_t i = 0;i<_size;++i){
    if(std::isinf(_array[i]) || std::isnan(_array[i])){
	value = true;
	break;
    }
       
  }

  return value;
}

void inplace_cpu_convolution(imageType* im,
			     int* imDim,
			     imageType* kernel,
			     int* kernelDim,
			     int nthreads){

  
  
  unsigned image_dim[3];
  unsigned kernel_dim[3];

  std::copy(imDim, imDim + 3, &image_dim[0]);
  std::copy(kernelDim, kernelDim + 3, &kernel_dim[0]);
  default_convolution convolver(im, image_dim, kernel, kernel_dim);

  if(nthreads!=1){
    parallel_transform::set_n_threads(nthreads);
    convolver.inplace<parallel_transform>();
  }
  else{
    convolver.inplace<serial_transform>();
  }

}


//implements http://arxiv.org/abs/1308.0730 (Eq 70)
void serial_inplace_cpu_deconvolve_iteration(imageType* psi,
					     workspace input,
					     double lambda, 
					     imageType minValue){

  std::vector<unsigned> image_dim(3);
  std::copy(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3, &image_dim[0]);

  multiviewnative::image_stack_ref input_psi(psi, image_dim);
  multiviewnative::image_stack integral = input_psi;

  view_data view_access;
  std::vector<unsigned> kernel1_dim(3);
  std::vector<unsigned> kernel2_dim(3);

  for(unsigned view = 0;view < input.num_views;++view){

    view_access = input.data_[view];

    std::copy(view_access.image_dims_    ,  view_access.image_dims_    +  3  ,  image_dim  .begin()  );
    std::copy(view_access.kernel1_dims_  ,  view_access.kernel1_dims_  +  3  ,  kernel1_dim.begin()  );
    std::copy(view_access.kernel2_dims_  ,  view_access.kernel2_dims_  +  3  ,  kernel2_dim.begin()  );

    integral = input_psi;
    //convolve: psi x kernel1 -> psiBlurred :: (Psi*P_v)
    default_convolution convolver1(integral.data(), &image_dim[0], view_access.kernel1_ , &kernel1_dim[0]);
    convolver1.inplace<serial_transform>();
    
    //view / psiBlurred -> psiBlurred :: (phi_v / (Psi*P_v))
    computeQuotient(view_access.image_,integral.data(),input_psi.num_elements());

    //convolve: psiBlurred x kernel2 -> integral :: (phi_v / (Psi*P_v)) * P_v^{compound}
    default_convolution convolver2(integral.data(), &image_dim[0], view_access.kernel2_, &kernel2_dim[0]);
    convolver2.inplace<serial_transform>();

    //computeFinalValues(input_psi,integral,weights)
    //studied impact of different techniques on how to implement this decision (decision in object, decision in if clause)
    //compiler opt & branch prediction seems to suggest this solution 
    if(lambda>0) 
      serial_regularized_final_values(input_psi.data(), integral.data(), view_access.weights_, 
				      input_psi.num_elements(),
				      lambda ,
				      minValue );
    else
      serial_final_values(input_psi.data(), integral.data(), view_access.weights_, 
			  input_psi.num_elements(),
			  minValue);
    
  }

}

//implements http://arxiv.org/abs/1308.0730 (Eq 70) using multiple threads
void parallel_inplace_cpu_deconvolve_iteration(imageType* psi,
					       workspace input,
					       int nthreads, 
					       double lambda, 
					       imageType minValue){

  std::vector<unsigned> image_dim(3);
  std::copy(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3, &image_dim[0]);

  multiviewnative::image_stack_ref input_psi(psi, image_dim);
  multiviewnative::image_stack integral = input_psi;

  view_data view_access;
  std::vector<unsigned> kernel1_dim(3);
  std::vector<unsigned> kernel2_dim(3);

  parallel_transform::set_n_threads(nthreads);

  for(unsigned view = 0;view < input.num_views;++view){

    view_access = input.data_[view];

    std::copy(view_access.image_dims_    ,  view_access.image_dims_    +  3  ,  image_dim  .begin()  );
    std::copy(view_access.kernel1_dims_  ,  view_access.kernel1_dims_  +  3  ,  kernel1_dim.begin()  );
    std::copy(view_access.kernel2_dims_  ,  view_access.kernel2_dims_  +  3  ,  kernel2_dim.begin()  );

    integral = input_psi;
    //convolve: psi x kernel1 -> psiBlurred :: (Psi*P_v)
    default_convolution convolver1(integral.data(), &image_dim[0], view_access.kernel1_ , &kernel1_dim[0]);
    convolver1.inplace<parallel_transform>();
    
    //view / psiBlurred -> psiBlurred :: (phi_v / (Psi*P_v))
    parallel_divide(view_access.image_,integral.data(),input_psi.num_elements(), nthreads);

    //convolve: psiBlurred x kernel2 -> integral :: (phi_v / (Psi*P_v)) * P_v^{compound}
    default_convolution convolver2(integral.data(), &image_dim[0], view_access.kernel2_, &kernel2_dim[0]);
    convolver2.inplace<parallel_transform>();

    //computeFinalValues(input_psi,integral,weights)
    //studied impact of different techniques on how to implement this decision (decision in object, decision in if clause)
    //compiler opt & branch prediction seems to suggest this solution
    if(lambda>0) 
      parallel_regularized_final_values(input_psi.data(), integral.data(), view_access.weights_, 
					input_psi.num_elements(),
					lambda,
					nthreads,
					minValue );
    else
      parallel_final_values(input_psi.data(), integral.data(), view_access.weights_, 
			    input_psi.num_elements(),
			    nthreads,
			    minValue);
  }

}



void inplace_cpu_deconvolve_iteration(imageType* psi,
				      workspace input,
				      int nthreads, 
				      double lambda, 
				      imageType minValue){

  if(nthreads == 1)
    serial_inplace_cpu_deconvolve_iteration(psi,input,lambda,minValue);
  else
    parallel_inplace_cpu_deconvolve_iteration(psi,input,nthreads,lambda,minValue);
}

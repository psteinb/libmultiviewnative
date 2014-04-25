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


void inplace_cpu_deconvolve_iteration(imageType* psi,
				      workspace input,
				      int nthreads, 
				      double lambda, 
				      imageType minValue){

  std::vector<unsigned> image_dim(3);
  std::copy(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3, &image_dim[0]);

  multiviewnative::image_stack_ref initial_psi(psi, image_dim);
  multiviewnative::image_stack current_psi = initial_psi;
  multiviewnative::image_stack integral = initial_psi;

  view_data view_access;
  std::vector<unsigned> kernel1_dim(3);
  std::vector<unsigned> kernel2_dim(3);

  for(unsigned view = 0;view < input.num_views;++view){

    view_access = input.data_[view];

    std::copy(view_access.image_dims_    ,  view_access.image_dims_    +  3  ,  image_dim  .begin()  );
    std::copy(view_access.kernel1_dims_  ,  view_access.kernel1_dims_  +  3  ,  kernel1_dim.begin()  );
    std::copy(view_access.kernel2_dims_  ,  view_access.kernel2_dims_  +  3  ,  kernel2_dim.begin()  );

if(check_nan(current_psi.data(),current_psi.num_elements())){
      std::cerr << "[before convolver1] current_psi contains nan! view: " << view << "\n";
break;
}

    //convolve: psi x kernel1 -> psiBlurred
    default_convolution convolver1(current_psi.data(), &image_dim[0], view_access.kernel1_ , &kernel1_dim[0]);
    convolver1.inplace<serial_transform>();

if(check_nan(current_psi.data(),current_psi.num_elements())){
      std::cerr << "[after convolver1] current_psi contains nan! view: " << view << "\n";
break;
}
    //image / psiBlurred -> psiBlurred 
    computeQuotient(view_access.image_,current_psi.data(),current_psi.num_elements());

if(check_nan(current_psi.data(),current_psi.num_elements())){
      std::cerr << "[after computeQuotient] current_psi contains nan! view: " << view << "\n";
break;
}
    //convolve: psiBlurred x kernel2 -> integral = current_psi
    integral = current_psi;
    default_convolution convolver2(integral.data(), &image_dim[0], view_access.kernel2_, &kernel2_dim[0]);
    convolver2.inplace<serial_transform>();

if(check_nan(integral.data(),integral.num_elements())){
      std::cerr << "[after convolver2] integral contains nan! view: " << view << "\n";
break;
}
    //computeFinalValues(initial_psi,integral,weights)
    computeFinalValues(current_psi.data(), integral.data(), view_access.weights_, 
		       current_psi.num_elements(),
		       0, lambda, minValue);

if(check_nan(current_psi.data(),current_psi.num_elements())){
      std::cerr << "[after computeFV] current_psi contains nan! view: " << view << "\n";
break;}
  }

  //running psi into initial array
  initial_psi = current_psi;
}

#define __MULTIVIEWNATIVE_CPP__

#include <vector>
#include <cmath>

#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"

#include "cpu_kernels.h"

namespace mvn = multiviewnative;

typedef mvn::cpu_convolve<> default_convolution;
typedef mvn::cpu_convolve<mvn::parallel_inplace_3d_transform> parallel_convolution;
typedef fftw_api_definitions<float> fftwf_api;

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

/**
   \brief inplace cpu based convolution, decides upon input value of nthreads whether to use single-threaded or multi-threaded implementation
   
   \param[in] im 1D array that contains the data image stack
   \param[in] imDim 3D array that contains the shape of the image stack im
   \param[in] kernel 1D array that contains the data kernel stack
   \param[in] kernelDim 3D array that contains the shape of the kernel stack kernel
   \param[in] nthreads number of threads to use
   
   \return 
   \retval 
   
*/
void inplace_cpu_convolution(imageType* im,
			     int* imDim,
			     imageType* kernel,
			     int* kernelDim,
			     int nthreads){

  
  
  unsigned image_dim[3];
  unsigned kernel_dim[3];

  std::copy(imDim, imDim + 3, &image_dim[0]);
  std::copy(kernelDim, kernelDim + 3, &kernel_dim[0]);
  
  if(nthreads!=1){
    parallel_convolution convolver(im, image_dim, kernel, kernel_dim); 
    parallel_convolution::transform_policy::set_n_threads(nthreads);
    convolver.inplace();
  }
  else{
    default_convolution convolver(im, image_dim, kernel, kernel_dim);
    convolver.inplace();
  }

}


//implements http://arxiv.org/abs/1308.0730 (Eq 70)
void serial_inplace_cpu_deconvolve_iteration(imageType* psi,
					     workspace input,
					     double lambda, 
					     imageType minValue){


  mvn::shape_t image_dim(3);
  std::copy(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3, &image_dim[0]);

  mvn::image_stack_ref input_psi(psi, image_dim);
  mvn::image_stack integral = input_psi;

  view_data view_access;
  mvn::shape_t kernel1_dim(3);
  mvn::shape_t kernel2_dim(3);

  
  for(unsigned view = 0;view < input.num_views_;++view){

    view_access = input.data_[view];

    std::copy(view_access.image_dims_    ,  view_access.image_dims_    +  3  ,  image_dim  .begin()  );
    std::copy(view_access.kernel1_dims_  ,  view_access.kernel1_dims_  +  3  ,  kernel1_dim.begin()  );
    std::copy(view_access.kernel2_dims_  ,  view_access.kernel2_dims_  +  3  ,  kernel2_dim.begin()  );

    integral = input_psi;
    //convolve: psi x kernel1 -> psiBlurred :: (Psi*P_v)
    default_convolution convolver1(integral.data(), &image_dim[0]);
    convolver1.inplace_with_forward_kernel(view_access.kernel1_ , &kernel1_dim[0]);
    
    //view / psiBlurred -> psiBlurred :: (phi_v / (Psi*P_v))
    computeQuotient(view_access.image_,integral.data(),input_psi.num_elements());

    //convolve: psiBlurred x kernel2 -> integral :: (phi_v / (Psi*P_v)) * P_v^{compound}
    default_convolution convolver2(integral.data(), &image_dim[0]);
    convolver2.inplace(view_access.kernel2_, &kernel2_dim[0]);

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

//implements http://arxiv.org/abs/1308.0730 (Eq 70)
void serial_inplace_cpu_deconvolve(imageType* psi,
				   workspace input,
				   double lambda, 
				   imageType minValue){
  
  //lay the kernel pointers aside
  std::vector<mvn::image_stack_ref> kernel1_ptr(input.num_views_);
  std::vector<mvn::image_stack_ref> kernel2_ptr(input.num_views_);
  std::vector<mvn::shape_t> image_shapes(input.num_views_);
  
  for( int v = 0;v<input.num_views_;++v){
    mvn::shape_t k1_dim(input.data->kernel1_dims_[v], input.data->kernel1_dims_[v] + mvn::image_stack_ref::dimensionality);
    kernel1_ptr[v] = image_stack_ref(input.data->kernel1_[v], k1_dim);

    mvn::shape_t k2_dim(input.data->kernel2_dims_[v], input.data->kernel2_dims_[v] + mvn::image_stack_ref::dimensionality);
    kernel2_ptr[v] = image_stack_ref(input.data->kernel2_[v], k2_dim);
  }

  //create the kernels in memory (this will double the memory consumption)
  std::vector<mvn::fftw_image_stack> forwarded_kernel1(input.num_views_);
  std::vector<mvn::fftw_image_stack> forwarded_kernel2(input.num_views_);
  
  
  for( int v = 0;v<input.num_views_;++v){

    image_shapes[v] = mvn::shape_t(&input.data.image_dims_[v], &input.data.image_dims_[v] + mvn::image_stack_ref::dimensionality);
    default_convolution::transform_policy fft(image_shapes[v]);

    default_convolution::padding_policy k1_padder(&image_shapes[v],kernel1_ptr[v].shape());
    default_convolution::padding_policy k2_padder(&image_shapes[v],kernel2_ptr[v].shape());

    //prepare the kernels for fft forward transform
    forwarded_kernel1[v].resize(image_shapes[v]);
    k1_padder.wrapped_insert_at_offsets(kernel1_ptr[v],forwarded_kernel1[v]);
    fft.padd_for_fft(forwarded_kernel1[v]);

    forwarded_kernel2[v].resize(image_shapes[v]);
    k2_padder.wrapped_insert_at_offsets(kernel2_ptr[v],forwarded_kernel2[v]);
    fft.padd_for_fft(forwarded_kernel2[v]);
    
  }
  
  //created and call batched fftw plan
  fftwf_api::plan_type k1_plan = fftwf_api::dft_r2c_many(mvn::image_stack_ref::dimensionality, //rank
							 (const int*)&image_shapes[0], //n
							 input.num_views_,//howmany
							 forwarded_kernel1[0].data(),//in
							 1, //istride
							 
							 
							 );
  fftwf_api::plan_type k2_plan = fftwf_api::dft_r2c_many(mvn::image_stack_ref::dimensionality);
  
  //put kernel pointers back
  for( int v = 0;v<input.num_views_;++v){
    input.data->kernel1_[v] = kernel1_ptr[v].data();
    input.data->kernel2_[v] = kernel2_ptr[v].data();
  }

}

//implements http://arxiv.org/abs/1308.0730 (Eq 70) using multiple threads
void parallel_inplace_cpu_deconvolve_iteration(imageType* psi,
					       workspace input,
					       int nthreads, 
					       double lambda, 
					       imageType minValue){

  mvn::shape_t image_dim(3);
  std::copy(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3, &image_dim[0]);

  mvn::image_stack_ref input_psi(psi, image_dim);
  mvn::image_stack integral = input_psi;

  view_data view_access;
  mvn::shape_t kernel1_dim(3);
  mvn::shape_t kernel2_dim(3);

  parallel_convolution::transform_policy::set_n_threads(nthreads);

  for(unsigned view = 0;view < input.num_views_;++view){

    view_access = input.data_[view];

    std::copy(view_access.image_dims_    ,  view_access.image_dims_    +  3  ,  image_dim  .begin()  );
    std::copy(view_access.kernel1_dims_  ,  view_access.kernel1_dims_  +  3  ,  kernel1_dim.begin()  );
    std::copy(view_access.kernel2_dims_  ,  view_access.kernel2_dims_  +  3  ,  kernel2_dim.begin()  );

    integral = input_psi;
    //convolve: psi x kernel1 -> psiBlurred :: (Psi*P_v)
    parallel_convolution convolver1(integral.data(), &image_dim[0], view_access.kernel1_ , &kernel1_dim[0]);
    convolver1.inplace();
    
    //view / psiBlurred -> psiBlurred :: (phi_v / (Psi*P_v))
    parallel_divide(view_access.image_,integral.data(),input_psi.num_elements(), nthreads);

    //convolve: psiBlurred x kernel2 -> integral :: (phi_v / (Psi*P_v)) * P_v^{compound}
    parallel_convolution convolver2(integral.data(), &image_dim[0], view_access.kernel2_, &kernel2_dim[0]);
    convolver2.inplace();

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


//implements http://arxiv.org/abs/1308.0730 (Eq 70) using multiple threads
void parallel_inplace_cpu_deconvolve(imageType* psi,
				     workspace input,
				     int nthreads, 
				     double lambda, 
				     imageType minValue){
  
  //TODO: place holder
  for(int it = 0;it<input.num_iterations_;++it){
    parallel_inplace_cpu_deconvolve_iteration(psi,
					      input,
					      nthreads,
					      input.lambda_,
					      input.minValue_);
  }

}

void inplace_cpu_deconvolve_iteration(imageType* psi,
				      workspace input,
				      int nthreads){

  if(nthreads == 1)
    serial_inplace_cpu_deconvolve_iteration(psi,
					    input,
					    input.lambda_,
					    input.minValue_);
  else
    parallel_inplace_cpu_deconvolve_iteration(psi,
					      input,
					      nthreads,
					      input.lambda_,
					      input.minValue_);
}



void inplace_cpu_deconvolve(imageType* psi,
			    workspace input,
			    int nthreads){
  
  //launch deconvolution
  if(nthreads == 1)
    serial_inplace_cpu_deconvolve(psi,
				  input,
				  input.lambda_,
				  input.minValue_);
  else
    parallel_inplace_cpu_deconvolve(psi,
				    input,
				    nthreads,
				    input.lambda_,
				    input.minValue_
				    );


}

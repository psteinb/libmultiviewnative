#ifndef _GPU_DECONVOLVE_METHODS_H_
#define _GPU_DECONVOLVE_METHODS_H_

// ------- C++ ----------
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

// ------- CUDA ----------
#include "cufft.h"

// ------- Project ----------
#include "multiviewnative.h"
#include "cuda_helpers.cuh"
#include "cuda_kernels.cuh"
#include "gpu_convolve.cuh"
#include "cufft_utils.cuh"
#include "padd_utils.h"
#include "image_stack_utils.h"


template <typename padding_type>
void generate_forwarded_kernels(std::vector<multiviewnative::image_stack>& _result,
				workspace input, 
				int kernel_id = 1
				)
{
  if(_result.size()!=input.num_views_)
    _result.resize(input.num_views_);

  int * kernel_dims = 0;
  float * kernel = 0;
  std::vector<int> reshaped;

  for (unsigned v = 0;v < _result.size();++v){
    
    kernel_dims = (kernel_id == 1) ? input.data_[v].kernel1_dims_ : input.data_[v].kernel2_dims_;
    kernel = (kernel_id == 1) ? input.data_[v].kernel1_ : input.data_[v].kernel2_;
    
    multiviewnative::shape_t kernel_shape(kernel_dims, 
					  kernel_dims + multiviewnative::image_stack::dimensionality);
    
    multiviewnative::image_stack_ref kernel_ref(kernel, kernel_shape);
    padding_type padding(input.data_[v].image_dims_, kernel_dims);
    
    //resize to image size
    _result[v].resize(padding.extents_);
    padding.wrapped_insert_at_offsets(kernel_ref, _result[v]);

    //resize to cufft compliance
    reshaped = multiviewnative::gpu::cufft_r2c_shape(_result[v].shape(),_result[v].shape() + 3);
    _result[v].resize(reshaped);
    
    //pin memory
    HANDLE_ERROR(cudaHostRegister((void*)_result[v].data(), 
				  _result[v].num_elements()*sizeof(float),
				  cudaHostRegisterPortable));
  }

}

/**
   \brief inplace convolution on workspace interlieving host-device copies with
   computations as much as possible
   \details See cuda_memory.cuh for the classes to facilitate this

   \param[in] input workspace that contains all input images, kernels (1+2) and
   weights
   \param[out] psi 3D image stack that will contain the output (it is expected
   to contain some form of start value)
   \param[in] device CUDA device to use (see nvidia-smi for details)

   \return
   \retval

*/
template <typename padding_type, 
	  typename convolve_type,
	  typename transform_type>
void inplace_gpu_deconvolve_iteration_interleaved(imageType* psi,
                                                  workspace input, int device) {

  using namespace multiviewnative;
  
  const unsigned n_views = input.num_views_;

  std::vector<image_stack> forwarded_kernels1(n_views);
  std::vector<image_stack> forwarded_kernels2(n_views);

  //prepare kernels (padd for cufft)
  generate_forwarded_kernels<padding_type>(forwarded_kernels1,input,1);
  generate_forwarded_kernels<padding_type>(forwarded_kernels2,input,2);

  std::vector<convolve_type*> view_folds(n_views,0);
  std::vector<image_stack> weights(n_views);

  shape_t input_shape(input.data_[0].image_dims_,input.data_[0].image_dims_ + 3);
  shape_t fftready_shape(forwarded_kernels1[0].shape(), forwarded_kernels1[0].shape() + 3);
  unsigned long padded_size_byte = forwarded_kernels1[0].num_elements()*sizeof(imageType);

  //prepare image, weights
  for (unsigned v = 0; v < view_folds.size(); ++v) {
    view_folds[v] = new convolve_type(input.data_[v].image_,
				      input.data_[v].image_dims_,
				      input.data_[v].kernel1_dims_
				      );
    weights[v].resize(input_shape);
    std::copy(input.data_[v].weights_, input.data_[v].weights_ + weights[v].num_elements(),
	      weights[v].data());
    weights[v].resize(fftready_shape);
  }

  //prepare space on device
  std::vector<float*> src_buffers(2);

  for (unsigned count = 0; count < src_buffers.size(); ++count){
    HANDLE_ERROR(cudaMalloc((void**)&(src_buffers[count]), padded_size_byte));
  }
  
  
  gpu::batched_fft_async2plans(forwarded_kernels1, 
			       input_shape, 
			       src_buffers, false);
  gpu::batched_fft_async2plans(forwarded_kernels2, 
			       input_shape, 
			       src_buffers, false);

  //expand memory on device
  src_buffers.reserve(4);
  for (unsigned count = 0; count < 2; ++count){
    float* temp = 0;
    HANDLE_ERROR(cudaMalloc((void**)&(temp), padded_size_byte));
    src_buffers.push_back(temp);
  }

  std::vector<cudaStream_t*> streams(2 // TODO: number of copy engines
				     );
  for( unsigned count = 0;count < streams.size();++count ){
    streams[count] = new cudaStream_t;
    HANDLE_ERROR(cudaStreamCreateWithFlags(streams[count], cudaStreamNonBlocking));
  }

  //src_buffers is 4 items large
  //use 
  // 0 .. any content 
  // 1 .. any content (mostly kernels)
  // 2 .. integral
  // 3 .. psi
  //fix the indices here
  const int psi_ = 3;
  const int intgr_ = 2;
  

  image_stack_ref input_psi(psi, input_shape);
  image_stack psi_stack = input_psi;
  psi_stack.resize(fftready_shape);
  
  HANDLE_ERROR(cudaMemcpy(src_buffers[psi_],
			  psi_stack.data(), 
			  padded_size_byte,
			  cudaMemcpyHostToDevice
			  ));
  
  const unsigned fft_num_elements = forwarded_kernels1[0].num_elements();
  const unsigned eff_fft_num_elements = fft_num_elements / 2;

  unsigned Threads = 128;//optimize later
  unsigned Blocks = largestDivisor(eff_fft_num_elements, Threads);

  for ( int i = 0; i < input.num_iterations_; ++i){

    HANDLE_ERROR(cudaMemcpyAsync(src_buffers[1],
				 forwarded_kernels1[0].data(), 
				 padded_size_byte,
				 cudaMemcpyHostToDevice,
				 *streams[0]
				 ));

    for (unsigned v = 0; v < n_views; ++v) {

      //integral = psi
      HANDLE_ERROR(cudaMemcpy(src_buffers[intgr_],
			      src_buffers[psi_], 
			      padded_size_byte,
			      cudaMemcpyDeviceToDevice
			      ));

      //would load internal from host
      // convolve: psi x kernel1 -> psiBlurred :: (Psi*P_v)
      inplace_asynch_convolve_on_device_and_kick<transform_type>(src_buffers[intgr_], 
						 src_buffers[1],
						 &input_shape[0],
						 fft_num_elements,
						 streams,
						 //goes to stream 1, src_buffer 1
						 view_folds[v]->padded_image_->data()
						 );
      
      //get kernel2 into buffer0
      HANDLE_ERROR(cudaMemcpyAsync(src_buffers[0],
				 forwarded_kernels2[v].data(), 
				 padded_size_byte,
				 cudaMemcpyHostToDevice,
				 *streams[0]
				 ));

      // view / psiBlurred -> psiBlurred :: (phi_v / (Psi*P_v))
      device_divide << <Blocks, Threads, 0, *streams[1] >>>
          (src_buffers[1], src_buffers[intgr_], fft_num_elements);
      HANDLE_LAST_ERROR();

      // convolve: psiBlurred x kernel2 -> integral :: (phi_v / (Psi*P_v)) *
      // P_v^{compound}
      inplace_asynch_convolve_on_device_and_kick<transform_type>(src_buffers[intgr_], 
						 src_buffers[0],
						 &input_shape[0],
						 fft_num_elements,
						 streams,
						 //goes to stream 1, src_buffer 1
						 weights[v].data()
						 );
      
      // computeFinalValues(input_psi,integral,weights)
      // studied impact of different techniques on how to implement this
      // decision (decision in object, decision in if clause)
      // compiler opt & branch prediction seems to suggest this solution
      if (input.lambda_ > 0) {
        device_regularized_final_values <<<Blocks, Threads, 0 , *streams[1]>>>
	  (src_buffers[psi_], src_buffers[intgr_], src_buffers[1],
	   input.lambda_, input.minValue_, fft_num_elements);

      } else {
        device_final_values <<<Blocks, Threads, 0 , *streams[1]>>>
	  (src_buffers[psi_], src_buffers[intgr_], src_buffers[1],
	   input.minValue_, fft_num_elements);
      }
      HANDLE_LAST_ERROR();
	
      //TODO: can this be removed?
      HANDLE_ERROR(cudaDeviceSynchronize());
    }

    
  }

  //clean-up
  for (unsigned v = 0; v < n_views; ++v) {
    delete view_folds[v];
  }

  for (unsigned b = 0; b < src_buffers.size(); ++b) {
    HANDLE_ERROR(cudaFree(src_buffers[b]));
  }

  //convert all data to what it was
  psi_stack.resize(input_shape);
  
  //copy result
  input_psi = psi_stack;
    
}

/**
   \brief inplace convolution on workspace performing the entire computation on
   device
   \details All data is transferred onto the device first and then the
   computations are performed.
   See cuda_memory.cuh for the classes to facilitate memory transfers.

   \param[in] input workspace that contains all input images, kernels (1+2) and
   weights
   \param[out] psi 3D image stack that will contain the output (it is expected
   to contain some form of start value)
   \param[in] device CUDA device to use (see nvidia-smi for details)

   \return
   \retval

*/
template <typename padding_type,typename transform_type>
void inplace_gpu_deconvolve_iteration_all_on_device(imageType* psi,
                                                    workspace input,
                                                    int device) {
  HANDLE_ERROR(cudaSetDevice(device));

  std::vector<padding_type> padding(input.num_views_);

  std::vector<multiviewnative::image_stack*> padded_view(input.num_views_);
  std::vector<multiviewnative::image_stack*> padded_kernel1(input.num_views_);
  std::vector<multiviewnative::image_stack*> padded_kernel2(input.num_views_);
  std::vector<multiviewnative::image_stack*> padded_weights(input.num_views_);
  std::vector<size_t> device_memory_elements_required(input.num_views_);

  std::vector<unsigned> image_dim(3);
  std::copy(input.data_[0].image_dims_, input.data_[0].image_dims_ + 3,
            &image_dim[0]);
  std::vector<unsigned> kernel_dim(image_dim.size());
  std::vector<unsigned> cufft_inplace_extents(kernel_dim.size());

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // PREPARE THE DATA (INCL PADDING)
  //
  for (unsigned v = 0; v < input.num_views_; ++v) {

    padding[v] = padding_type(input.data_[v].image_dims_,
			      input.data_[v].kernel1_dims_);
    std::copy(input.data_[0].kernel1_dims_, input.data_[0].kernel1_dims_ + 3,
              &kernel_dim[0]);

    padded_view[v] = new multiviewnative::image_stack(padding[v].extents_);
    padded_weights[v] = new multiviewnative::image_stack(padding[v].extents_);
    padded_kernel1[v] = new multiviewnative::image_stack(padding[v].extents_);
    padded_kernel2[v] = new multiviewnative::image_stack(padding[v].extents_);

    multiviewnative::image_stack_cref view(input.data_[v].image_, image_dim);
    multiviewnative::image_stack_cref weights(input.data_[v].weights_,
                                              image_dim);
    multiviewnative::image_stack_cref kernel1(input.data_[v].kernel1_,
                                              kernel_dim);
    multiviewnative::image_stack_cref kernel2(input.data_[v].kernel2_,
                                              kernel_dim);

    padding[v].insert_at_offsets(view, *padded_view[v]);
    padding[v].insert_at_offsets(weights, *padded_weights[v]);
    padding[v].wrapped_insert_at_offsets(kernel1, *padded_kernel1[v]);
    padding[v].wrapped_insert_at_offsets(kernel2, *padded_kernel2[v]);

    multiviewnative::adapt_extents_for_fftw_inplace(
        padding[v].extents_, cufft_inplace_extents,
        padded_view[v]->storage_order());
    device_memory_elements_required[v] = std::accumulate(
        cufft_inplace_extents.begin(), cufft_inplace_extents.end(), 1,
        std::multiplies<size_t>());
  }

  multiviewnative::image_stack_ref input_psi(psi, image_dim);
  multiviewnative::image_stack padded_psi(padding[0].extents_);
  padding_type input_psi_padder = padding[0];
  input_psi_padder.insert_at_offsets(input_psi, padded_psi);
  unsigned long max_device_memory_elements_required =
      *std::max_element(device_memory_elements_required.begin(),
                        device_memory_elements_required.end());

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_view[0]->num_elements(), size_t(threads.x)));
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // ITERATE
  //
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_running_psi(
      padded_psi, max_device_memory_elements_required);
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_integral(
      max_device_memory_elements_required);
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_view(
      max_device_memory_elements_required);
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_kernel1(
      max_device_memory_elements_required);
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_kernel2(
      max_device_memory_elements_required);
  multiviewnative::stack_on_device<multiviewnative::image_stack> d_weights(
      max_device_memory_elements_required);

  unsigned long long current_gmem_usage_byte =
      6 * max_device_memory_elements_required;
  if (current_gmem_usage_byte > .25 * getAvailableGMemOnCurrentDevice()) {
    std::cout << "current gmem footprint ("
              << current_gmem_usage_byte / float(1 << 20)
              << " MB) exceeds available memory threshold: (free) "
              << getAvailableGMemOnCurrentDevice() / float(1 << 20)
              << " MB, threshold: " << .25 * getAvailableGMemOnCurrentDevice() /
                                           float(1 << 20) << " MB\n";
  }

  for (int iteration = 0; iteration < input.num_iterations_; ++iteration) {

    for (int v = 0; v < input.num_views_; ++v) {

      d_integral = d_running_psi;
      HANDLE_LAST_ERROR();
      d_kernel1.push_to_device(*padded_kernel1[v]);
      HANDLE_LAST_ERROR();
      // integral = integral * kernel1
      multiviewnative::inplace_convolve_on_device<transform_type>(
          d_integral.data(), d_kernel1.data(), &padding[v].extents_[0],
          device_memory_elements_required[v]);
      HANDLE_LAST_ERROR();

      d_view.push_to_device(*padded_view[v]);
      HANDLE_LAST_ERROR();
      device_divide << <blocks, threads>>>
          (d_view.data(), d_integral.data(), padded_view[v]->num_elements());
      HANDLE_LAST_ERROR();
      d_kernel2.push_to_device(*padded_kernel2[v]);
      HANDLE_LAST_ERROR();
      multiviewnative::inplace_convolve_on_device<transform_type>(
          d_integral.data(), d_kernel2.data(), &padding[v].extents_[0],
          device_memory_elements_required[v]);
      HANDLE_LAST_ERROR();
      d_weights.push_to_device(*padded_weights[v]);
      HANDLE_LAST_ERROR();
      if (input.lambda_ > 0) {
        device_regularized_final_values << <blocks, threads>>>
            (d_running_psi.data(), d_integral.data(), d_weights.data(),
             input.lambda_, input.minValue_, padded_view[v]->num_elements());

      } else {
        device_final_values << <blocks, threads>>>
            (d_running_psi.data(), d_integral.data(), d_weights.data(),
             input.minValue_, padded_view[v]->num_elements());
      }
      HANDLE_LAST_ERROR();
    }
  }

  d_running_psi.pull_from_device(padded_psi);

  input_psi = padded_psi
      [boost::indices[multiviewnative::range(
          input_psi_padder.offsets_[0],
          input_psi_padder.offsets_[0] + input_psi.shape()[0])]
                     [multiviewnative::range(
                         input_psi_padder.offsets_[1],
                         input_psi_padder.offsets_[1] + input_psi.shape()[1])]
                     [multiviewnative::range(
                         input_psi_padder.offsets_[2],
                         input_psi_padder.offsets_[2] + input_psi.shape()[2])]];

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // CLEAN-UP
  //
  for (int v = 0; v < input.num_views_; ++v) {

    delete padded_view[v];
    delete padded_kernel1[v];
    delete padded_kernel2[v];
    delete padded_weights[v];
  }
}


#endif /* _GPU_DECONVOLVE_METHODS_H_ */

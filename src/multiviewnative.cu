#define __MULTIVIEWNATIVE_CU__
// ------- C++ ----------
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

// ------- CUDA ----------
#include "cuda.h"
#include "cufft.h"

// ------- Project ----------
#include "multiviewnative.h"
#include "cuda_helpers.cuh"
#include "cuda_kernels.cuh"
#include "gpu_convolve.cuh"
#include "cufft_utils.cuh"
#include "padd_utils.h"
#include "image_stack_utils.h"

typedef multiviewnative::zero_padd<multiviewnative::image_stack>
    wrap_around_padding;

typedef multiviewnative::zero_padd<multiviewnative::image_stack>
    as_is_padding;

typedef multiviewnative::inplace_3d_transform_on_device<imageType>
    device_transform;

typedef multiviewnative::gpu_convolve<wrap_around_padding, imageType, unsigned>
    device_convolve;

//TODO:
//this is the convolution we wanna use in the end
typedef multiviewnative::gpu_convolve<as_is_padding, imageType, unsigned>
    target_convolve;

/**
   \brief Function to perform an inplace convolution (all inputs will received
   wrap_around_padding)

   \param[in] im 1D array that contains the data image stack
   \param[in] imDim 3D array that contains the shape of the image stack im
   \param[in] kernel 1D array that contains the data kernel stack
   \param[in] kernelDim 3D array that contains the shape of the kernel stack
   kernel
   \param[in] device CUDA device to use (see nvidia-smi for details)

   \return
   \retval

*/
void inplace_gpu_convolution(imageType* im, int* imDim, imageType* kernel,
                             int* kernelDim, int device) {

  using namespace multiviewnative;

  unsigned image_dim[3];
  unsigned kernel_dim[3];
  std::copy(imDim, imDim + 3, &image_dim[0]);
  std::copy(kernelDim, kernelDim + 3, &kernel_dim[0]);

  device_convolve convolver(im, image_dim, kernel, kernel_dim);

  if (device < 0) device = selectDeviceWithHighestComputeCapability();

  convolver.set_device(device);

  convolver.inplace<device_transform>();
}


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
void inplace_gpu_deconvolve_iteration_interleaved(imageType* psi,
                                                  workspace input, int device) {

  using namespace multiviewnative;
  
  const unsigned n_views = input.num_views_;

  std::vector<image_stack> forwarded_kernels1(n_views);
  std::vector<image_stack> forwarded_kernels2(n_views);

  //prepare kernels (padd for cufft)
  generate_forwarded_kernels<as_is_padding>(forwarded_kernels1,input,1);
  generate_forwarded_kernels<as_is_padding>(forwarded_kernels2,input,2);

  std::vector<target_convolve*> view_folds(n_views,0);
  std::vector<image_stack> weights(n_views);

  shape_t input_shape(input.data_[0].image_dims_,input.data_[0].image_dims_ + 3);
  shape_t common_shape(forwarded_kernels1[0].shape(), forwarded_kernels1[0].shape() + 3);
  unsigned long padded_size_byte = forwarded_kernels1[0].num_elements()*sizeof(imageType);

  //prepare image, weights
  for (unsigned v = 0; v < view_folds.size(); ++v) {
    view_folds[v] = new target_convolve(input.data_[v].image_,
					input.data_[v].image_dims_,
					input.data_[v].kernel1_dims_
					);
    weights[v].resize(input_shape);
    std::copy(input.data_[v].weights_, input.data_[v].weights_ + weights[v].num_elements(),
	      weights[v].data());
    weights[v].resize(common_shape);
  }

  //prepare space on device
  std::vector<float*> src_buffers(2);

  for (unsigned count = 0; count < src_buffers.size(); ++count){
    HANDLE_ERROR(cudaMalloc((void**)&(src_buffers[count]), padded_size_byte));
  }
  
  
  gpu::batched_fft_async2plans(forwarded_kernels1, common_shape, src_buffers, false);
  gpu::batched_fft_async2plans(forwarded_kernels2, common_shape, src_buffers, false);

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
    HANDLE_ERROR(cudaStreamCreate(streams[count]));
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
  psi_stack.resize(common_shape);
  
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
      inplace_asynch_convolve_on_device_and_kick<device_transform>(src_buffers[intgr_], 
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
      inplace_asynch_convolve_on_device_and_kick<device_transform>(src_buffers[intgr_], 
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
void inplace_gpu_deconvolve_iteration_all_on_device(imageType* psi,
                                                    workspace input,
                                                    int device) {
  HANDLE_ERROR(cudaSetDevice(device));

  std::vector<wrap_around_padding> padding(input.num_views_);

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

    padding[v] = wrap_around_padding(input.data_[v].image_dims_,
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
  wrap_around_padding input_psi_padder = padding[0];
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
      multiviewnative::inplace_convolve_on_device<device_transform>(
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
      multiviewnative::inplace_convolve_on_device<device_transform>(
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

/**
   \brief dispatch function, this one decides wether to try and do the entire
   computation on the device or not and then dispatches the appropriate call



   \return
   \retval

*/
void inplace_gpu_deconvolve(imageType* psi, workspace input, int device) {

  if (device < 0) device = selectDeviceWithHighestComputeCapability();


  long long device_gmem_byte = getMemDeviceCUDA(device);
  unsigned device_gmem_mb = device_gmem_byte >> 20;

  size_t cufft_workarea = 0;
  cufftEstimate3d(input.data_[0].image_dims_[0], input.data_[0].image_dims_[1],
                  input.data_[0].image_dims_[2], CUFFT_R2C, &cufft_workarea);
  HANDLE_LAST_ERROR();
  float cufft_workarea_mb = cufft_workarea/(1024*1024.);
  
  size_t single_stack_in_byte =
      sizeof(imageType) * std::accumulate(input.data_[0].image_dims_,
                                          input.data_[0].image_dims_ + 3, 1.,
                                          std::multiplies<int>());
  float single_stack_in_mb = single_stack_in_byte / (1024*1024.);

  float memory_fft_step_mb = 2*single_stack_in_mb + cufft_workarea_mb;
  float regularisation_step_mb = 3*single_stack_in_mb;
  float min_memory_budget_mb = std::max(regularisation_step_mb,memory_fft_step_mb);

  // decide if the incoming data fills the memory on device too much
  float memory_all_on_device_mb = (4*input.num_views_ + 2)*single_stack_in_mb + cufft_workarea_mb;
  

  // cufft is memory hungry, that is why we only push all stacks to device mem
  // if the total budget does not exceed 1/3 device mem
  bool all_on_device = memory_all_on_device_mb < (device_gmem_mb * .9);
  std::cout << "[lmvn::inplace_gpu_deconvolve] FFT: "
            << memory_all_on_device_mb << " MB (all-on-device), "
	    << min_memory_budget_mb << " MB (min-mem interleaved), "
            << " available on GPU: " << device_gmem_mb
            << " MB ... ";

  if (all_on_device){
    std::cout << "all on device!\n";
    inplace_gpu_deconvolve_iteration_all_on_device(psi, input, device);
    return;
  }

  if(min_memory_budget_mb < device_gmem_mb){
    std::cout << "interleaved!\n";
    inplace_gpu_deconvolve_iteration_interleaved(psi, input, device);
    return;
  }

  std::cerr << "[lmvn::inplace_gpu_deconvolve] FFT: Unable to run on GPU due to memory constraints!\n";
  
}

#ifndef LB_MAX_THREADS
#define LB_MAX_THREADS 1024
#endif

#ifndef DIMSIMAGE
static const int dimsImage = 3;
#else
static const int dimsImage = DIMSIMAGE;
#endif

__global__ void __launch_bounds__(LB_MAX_THREADS)
    fftShiftKernel(imageType* kernelCUDA, imageType* kernelPaddedCUDA,
                   unsigned int kernelDim_0, unsigned int kernelDim_1,
                   unsigned int kernelDim_2, unsigned int imDim_0,
                   unsigned int imDim_1, unsigned int imDim_2) {
  long int kernelSize = kernelDim_0 * kernelDim_1 * kernelDim_2;
  long int imageSize = imDim_0 * imDim_1 * imDim_2;

  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  long int x, y, z, aux;
  if (tid < kernelSize) {
    // find coordinates
    z = tid - (tid / kernelDim_2) * kernelDim_2;
    aux = (tid - z) / kernelDim_2;
    y = aux - (aux / kernelDim_1) * kernelDim_1;
    x = (aux - y) / kernelDim_1;

    // center coordinates
    x -= (long int)kernelDim_0 / 2;
    y -= (long int)kernelDim_1 / 2;
    z -= (long int)kernelDim_2 / 2;

    // circular shift if necessary
    if (x < 0) x += imDim_0;
    if (y < 0) y += imDim_1;
    if (z < 0) z += imDim_2;

    // WOW! this is a depth-major format
    // calculate position in padded kernel
    aux = z + imDim_2 * (y + imDim_1 * x);

    // copy value
    if (aux < imageSize)
      kernelPaddedCUDA[aux] = kernelCUDA[tid];  // for the most part it should
                                                // be a coalescent access in oth
                                                // places
  }
}

//=====================================================================
// WARNING: for cuFFT the fastest running index is z direction!!! so pos = z +
// imDim[2] * (y + imDim[1] * x)
// NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting
// factor
void convolution3DfftCUDAInPlace(imageType* im, int* imDim, imageType* kernel,
                                 int* kernelDim, int devCUDA) {
  imageType* imCUDA = NULL;
  imageType* kernelCUDA = NULL;

  HANDLE_ERROR(cudaSetDevice(devCUDA));

  size_t imSize = std::accumulate(imDim, imDim + 3, 1, std::multiplies<int>());
  size_t kernelSize =
      std::accumulate(kernelDim, kernelDim + 3, 1, std::multiplies<int>());

  size_t imSizeFFT = imSize;
  imSizeFFT +=
      2 * imDim[0] * imDim[1];  // size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT * sizeof(imageType);
  size_t imSizeInByte = imSize * sizeof(imageType);
  size_t kernelSizeInByte = (kernelSize) * sizeof(imageType);
  // allocat ememory in GPU
  HANDLE_ERROR(cudaMalloc((void**)&(imCUDA), imSizeFFTInByte));  // a little bit
                                                                 // larger to
                                                                 // allow
                                                                 // in-place FFT
  HANDLE_ERROR(cudaMalloc((void**)&(kernelCUDA), kernelSizeInByte));

  HANDLE_ERROR(
      cudaMemcpy(kernelCUDA, kernel, kernelSizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(imCUDA, im, imSizeInByte, cudaMemcpyHostToDevice));

  ///////////////////////////////////////////////////////////////////////
  convolution3DfftCUDAInPlace_core(imCUDA, imDim, kernelCUDA, kernelDim,
                                   devCUDA);
  ///////////////////////////////////////////////////////////////////////

  // copy result to host and overwrite image
  HANDLE_ERROR(cudaMemcpy(im, imCUDA, sizeof(imageType) * imSize,
                          cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(imCUDA));
  HANDLE_ERROR(cudaFree(kernelCUDA));
}

//=====================================================================
// WARNING: for cuFFT the fastest running index is z direction!!! so pos = z +
// imDim[2] * (y + imDim[1] * x)
// NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting
// factor
void convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA, int* imDim,
                                      imageType* _d_kernelCUDA, int* kernelDim,
                                      int devCUDA) {
  cufftHandle fftPlanFwd, fftPlanInv;
  imageType* kernelPaddedCUDA = NULL;

  size_t imSize = 1;
  size_t kernelSize = 1;
  for (int ii = 0; ii < dimsImage; ii++) {
    imSize *= (imDim[ii]);
    kernelSize *= (kernelDim[ii]);
  }

  size_t imSizeFFT = imSize;
  imSizeFFT += 2 * imDim[0] * imDim[1];
  size_t imSizeFFTInByte = imSizeFFT * sizeof(imageType);

  HANDLE_ERROR(cudaMalloc((void**)&(kernelPaddedCUDA), imSizeFFTInByte));
  HANDLE_ERROR(cudaMemset(kernelPaddedCUDA, 0, imSizeFFTInByte));

  size_t max_threads_on_device = getMaxNThreadsOfDevice(devCUDA);
  size_t max_blocks_in_x =
      getMaxNBlocksOfDevice(devCUDA, 0);  // we are using dim1 blocks only

  int numThreads = std::min(max_threads_on_device, kernelSize);
  size_t numBlocksFromImage = (kernelSize + numThreads - 1) / (numThreads);
  int numBlocks = std::min(max_blocks_in_x, numBlocksFromImage);

  fftShiftKernel << <numBlocks, numThreads>>>
      (_d_kernelCUDA, kernelPaddedCUDA, kernelDim[0], kernelDim[1],
       kernelDim[2], imDim[0], imDim[1], imDim[2]);
  HANDLE_ERROR_KERNEL;

  // make sure GPU finishes
  HANDLE_ERROR(cudaDeviceSynchronize());

  cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C);
  HANDLE_ERROR_KERNEL;
  cufftSetCompatibilityMode(fftPlanFwd, CUFFT_COMPATIBILITY_NATIVE);
  HANDLE_ERROR_KERNEL;  // for highest performance since we do not need FFTW
                        // compatibility

  // inPlace FFT for image and kernel
  cufftExecR2C(fftPlanFwd, _d_imCUDA, (cufftComplex*)_d_imCUDA);
  HANDLE_ERROR_KERNEL;
  cufftExecR2C(fftPlanFwd, kernelPaddedCUDA, (cufftComplex*)kernelPaddedCUDA);
  HANDLE_ERROR_KERNEL;

  size_t halfImSizeFFT = imSizeFFT / 2;
  numThreads = std::min(max_threads_on_device, halfImSizeFFT);
  numBlocksFromImage = (halfImSizeFFT + numThreads - 1) / (numThreads);
  numBlocks = std::min(max_blocks_in_x, numBlocksFromImage);

  // convolve
  float scale = 1.0f / float(imSize);
  modulateAndNormalize_kernel << <numBlocks, numThreads>>>
      ((cufftComplex*)_d_imCUDA, (cufftComplex*)kernelPaddedCUDA, halfImSizeFFT,
       scale);
  HANDLE_ERROR_KERNEL;  // last parameter is the size of the FFT

  // inverse FFT of image only
  cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);
  HANDLE_ERROR_KERNEL;
  cufftSetCompatibilityMode(fftPlanInv, CUFFT_COMPATIBILITY_NATIVE);
  HANDLE_ERROR_KERNEL;
  cufftExecC2R(fftPlanInv, (cufftComplex*)_d_imCUDA, _d_imCUDA);
  HANDLE_ERROR_KERNEL;

  // release memory
  HANDLE_ERROR(cudaFree(kernelPaddedCUDA));
  (cufftDestroy(fftPlanFwd));
  HANDLE_ERROR_KERNEL;
  (cufftDestroy(fftPlanInv));
  HANDLE_ERROR_KERNEL;
}

void compute_quotient(imageType* _input, imageType* _output, size_t _size,
                      int _device) {

  imageType* d_input = 0;
  imageType* d_output = 0;

  const size_t sizeInByte = _size * sizeof(imageType);
  HANDLE_ERROR(cudaSetDevice(_device));

  HANDLE_ERROR(cudaMalloc((void**)&d_input, sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_output, sizeInByte));

  HANDLE_ERROR(cudaMemcpy(d_input, _input, sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_output, _output, sizeInByte, cudaMemcpyHostToDevice));

  size_t items_per_block = 128;
  size_t items_per_grid = _size / items_per_block;

  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid);

  fit_2Dblocks_to_threads_for_device(threads, blocks, _device);

  // performs d_input_[:]=d_output_[:]/d_input_[:]
  device_divide << <blocks, threads>>> (d_input, d_output, (unsigned int)_size);

  HANDLE_ERROR(
      cudaMemcpy(_output, d_output, sizeInByte, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(d_input));
  HANDLE_ERROR(cudaFree(d_output));
}

void compute_final_values(imageType* _image, imageType* _integral,
                          imageType* _weight, size_t _size, float _minValue,
                          double _lambda, int _device) {

  imageType* d_integral = 0;
  imageType* d_weight = 0;
  imageType* d_image = 0;

  const size_t sizeInByte = _size * sizeof(imageType);

  HANDLE_ERROR(cudaMalloc((void**)&d_integral, sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_weight, sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_image, sizeInByte));

  HANDLE_ERROR(
      cudaMemcpy(d_integral, _integral, sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_weight, _weight, sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_image, _image, sizeInByte, cudaMemcpyHostToDevice));

  size_t items_per_block = 128;
  size_t items_per_grid = _size / items_per_block;

  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid);
  fit_2Dblocks_to_threads_for_device(threads, blocks, _device);

  if (_lambda > 0.)
    device_regularized_final_values << <blocks, threads>>>
        (d_image, d_integral, d_weight, _lambda, _minValue, _size);
  else
    device_final_values << <blocks, threads>>>
        (d_image, d_integral, d_weight, _minValue, _size);

  HANDLE_ERROR(cudaMemcpy(_image, d_image, sizeInByte, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(d_integral));
  HANDLE_ERROR(cudaFree(d_weight));
  HANDLE_ERROR(cudaFree(d_image));
}

void iterate_fft_plain(imageType* _input, imageType* _kernel,
                       imageType* _output, int* _input_dims, int* _kernel_dims,
                       int _device) {

  size_t inputSize = std::accumulate(
      &_input_dims[0], &_input_dims[0] + dimsImage, 1, multiplies<int>());
  size_t kernelSize = std::accumulate(
      &_kernel_dims[0], &_kernel_dims[0] + dimsImage, 1, multiplies<int>());

  size_t inputInByte = inputSize * sizeof(imageType);
  size_t kernelInByte = kernelSize * sizeof(imageType);

  imageType* input = 0;
  imageType* kernel1 = 0;
  imageType* weights = 0;
  imageType* kernel2 = 0;
  HANDLE_ERROR(
      cudaMallocHost((void**)&(kernel2), kernelInByte, cudaHostAllocDefault));
  HANDLE_ERROR(
      cudaMallocHost((void**)&(weights), inputInByte, cudaHostAllocDefault));
  HANDLE_ERROR(
      cudaMallocHost((void**)&(kernel1), kernelInByte, cudaHostAllocDefault));
  HANDLE_ERROR(
      cudaMallocHost((void**)&(input), inputInByte, cudaHostAllocDefault));

  std::fill(&kernel2[0], &kernel2[0] + kernelSize, .1f);
  std::fill(&weights[0], &weights[0] + inputSize, 1.f);
  std::copy(&_kernel[0], &_kernel[0] + kernelSize, &kernel1[0]);
  std::copy(&_input[0], &_input[0] + inputSize, &input[0]);

  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //
  imageType* d_image_ = 0;
  imageType* d_initial_ = 0;
  imageType* d_kernel_ = 0;
  imageType* d_weights_ = 0;

  size_t imSizeFFT =
      inputSize +
      2 * _input_dims[0] *
          _input_dims[1];  // size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT * sizeof(imageType);
  int gpu_device = selectDeviceWithHighestComputeCapability();

  HANDLE_ERROR(cudaMalloc((void**)&(d_image_), imSizeFFTInByte));  // a little
                                                                   // bit larger
                                                                   // to allow
                                                                   // in-place
                                                                   // FFT
  HANDLE_ERROR(cudaMalloc((void**)&(d_initial_), inputInByte));
  HANDLE_ERROR(cudaMalloc((void**)&(d_weights_), inputInByte));
  HANDLE_ERROR(cudaMalloc((void**)&(d_kernel_), kernelInByte));
  cudaStream_t initial_stream, weights_stream;
  HANDLE_ERROR(cudaStreamCreate(&initial_stream));
  HANDLE_ERROR(cudaStreamCreate(&weights_stream));

  // TODO: should the weights be updated from device_divide (unclear in the java
  // application)?
  HANDLE_ERROR(cudaMemcpyAsync(d_weights_, weights, inputInByte,
                               cudaMemcpyHostToDevice, weights_stream));
  HANDLE_ERROR(cudaMemcpyAsync(d_initial_, input, inputInByte,
                               cudaMemcpyHostToDevice, initial_stream));

  HANDLE_ERROR(
      cudaMemcpy(d_kernel_, kernel1, kernelInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_image_, input, inputInByte, cudaMemcpyHostToDevice));

  // convolve(input) with kernel1 -> psiBlurred
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims, d_kernel_,
                                   _kernel_dims, gpu_device);

  // computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = inputSize / items_per_block;

  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid);
  fit_2Dblocks_to_threads_for_device(threads, blocks, gpu_device);

  // performs d_initial_[:]=d_image_[:]/d_initial_[:]
  // TODO: should the weights be updated?
  device_divide << <blocks, threads, 0, initial_stream>>>
      (d_initial_, d_image_, inputSize);

  // convolve(psiBlurred) with kernel2 -> integral
  HANDLE_ERROR(
      cudaMemcpy(d_kernel_, kernel2, kernelInByte, cudaMemcpyHostToDevice));
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims, d_kernel_,
                                   _kernel_dims, gpu_device);
  // computeFinalValues(input,integral,weights)
  device_final_values << <blocks, threads, 0, weights_stream>>>
      (d_initial_, d_image_, d_weights_, .0001f, inputSize);

  HANDLE_ERROR(cudaMemcpyAsync(_output, d_initial_, inputInByte,
                               cudaMemcpyDeviceToHost, weights_stream));
  HANDLE_ERROR(cudaStreamSynchronize(weights_stream));

  HANDLE_ERROR(cudaFree(d_image_));
  HANDLE_ERROR(cudaFree(d_initial_));
  HANDLE_ERROR(cudaFree(d_kernel_));
  HANDLE_ERROR(cudaFree(d_weights_));

  HANDLE_ERROR(cudaFreeHost(kernel2));
  HANDLE_ERROR(cudaFreeHost(weights));
  HANDLE_ERROR(cudaFreeHost(kernel1));
  HANDLE_ERROR(cudaFreeHost(input));
  HANDLE_ERROR(cudaStreamDestroy(initial_stream));
  HANDLE_ERROR(cudaStreamDestroy(weights_stream));
}

void iterate_fft_tikhonov(imageType* _input, imageType* _kernel,
                          imageType* _output, int* _input_dims,
                          int* _kernel_dims, size_t _size, float _minValue,
                          double _lambda, int _device) {

  size_t inputSize = std::accumulate(
      &_input_dims[0], &_input_dims[0] + dimsImage, 1, multiplies<int>());
  size_t kernelSize = std::accumulate(
      &_kernel_dims[0], &_kernel_dims[0] + dimsImage, 1, multiplies<int>());

  size_t inputInByte = inputSize * sizeof(imageType);
  size_t kernelInByte = kernelSize * sizeof(imageType);

  std::vector<imageType>* kernel2_ = new std::vector<imageType>(kernelSize);
  std::vector<imageType>* weights_ = new std::vector<imageType>(inputSize);
  std::fill(kernel2_->begin(), kernel2_->end(), .1f);
  std::fill(weights_->begin(), weights_->end(), 1.f);

  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //
  imageType* d_image_ = 0;
  imageType* d_initial_ = 0;
  imageType* d_kernel_ = 0;
  imageType* d_weights_ = 0;

  size_t imSizeFFT =
      inputSize +
      2 * _input_dims[0] *
          _input_dims[1];  // size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT * sizeof(imageType);
  int gpu_device = selectDeviceWithHighestComputeCapability();

  HANDLE_ERROR(cudaMalloc((void**)&(d_image_), imSizeFFTInByte));  // a little
                                                                   // bit larger
                                                                   // to allow
                                                                   // in-place
                                                                   // FFT
  HANDLE_ERROR(cudaMalloc((void**)&(d_initial_), inputInByte));
  HANDLE_ERROR(cudaMalloc((void**)&(d_weights_), inputInByte));
  HANDLE_ERROR(cudaMalloc((void**)&(d_kernel_), kernelInByte));
  HANDLE_ERROR(cudaMemcpy(d_weights_, &weights_[0], inputInByte,
                          cudaMemcpyHostToDevice));

  HANDLE_ERROR(
      cudaMemcpy(d_kernel_, _kernel, kernelInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_image_, _input, inputInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(d_initial_, d_image_, inputInByte, cudaMemcpyDeviceToDevice));

  // convolve(input) with kernel1 -> psiBlurred
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims, d_kernel_,
                                   _kernel_dims, gpu_device);

  // computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = inputSize / items_per_block;

  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid);
  fit_2Dblocks_to_threads_for_device(threads, blocks, gpu_device);

  // performs d_initial_[:]=d_image_[:]/d_initial_[:]
  // TODO: should the weights be updated?
  device_divide << <blocks, threads>>> (d_initial_, d_image_, inputSize);

  // convolve(psiBlurred) with kernel2 -> integral
  HANDLE_ERROR(cudaMemcpy(d_kernel_, &kernel2_[0], kernelInByte,
                          cudaMemcpyHostToDevice));
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims, d_kernel_,
                                   _kernel_dims, gpu_device);
  // computeFinalValues(input,integral,weights)
  device_finalValues_tikhonov << <blocks, threads>>>
      (d_initial_, d_image_, d_weights_, .0001f, .2f, inputSize);

  HANDLE_ERROR(
      cudaMemcpy(_output, d_initial_, inputInByte, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(d_image_));
  HANDLE_ERROR(cudaFree(d_initial_));
  HANDLE_ERROR(cudaFree(d_kernel_));
  HANDLE_ERROR(cudaFree(d_weights_));

  delete kernel2_;
  delete weights_;
}

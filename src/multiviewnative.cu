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

#ifndef LB_MAX_THREADS
#define LB_MAX_THREADS 1024 
#endif

#ifndef DIMSIMAGE
static const int dimsImage = 3;
#else
static const int dimsImage = DIMSIMAGE;
#endif

__global__ void  __launch_bounds__(LB_MAX_THREADS) fftShiftKernel(imageType* kernelCUDA,
								       imageType* kernelPaddedCUDA,
								       unsigned int kernelDim_0,
								       unsigned int kernelDim_1,
								       unsigned int kernelDim_2,
								       unsigned int imDim_0,
								       unsigned int imDim_1,
								       unsigned int imDim_2)
{
	int kernelSize = kernelDim_0 * kernelDim_1 * kernelDim_2;

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int x,y,z;
	unsigned aux;
	if(tid<kernelSize)
	  {
	    //find coordinates
	    z = tid - (tid / kernelDim_2);
	    aux = (tid - z)/kernelDim_2;
	    y = aux - (aux / kernelDim_1);
	    x = (aux - y)/kernelDim_1;

	    //center coordinates
	    x -= (int)kernelDim_0/2;
	    y -= (int)kernelDim_1/2;
	    z -= (int)kernelDim_2/2;

	    //circular shift if necessary
	    if(x<0) x += imDim_0;
	    if(y<0) y += imDim_1;
	    if(z<0) z += imDim_2;

	    //WOW! this is a depth-major format
	    //calculate position in padded kernel
	    aux = (unsigned int)z + imDim_2 * ((unsigned int)y + imDim_1 * (unsigned int)x);

	    //copy value
	    if(aux<(imDim_0 * imDim_1 * imDim_2))
	      kernelPaddedCUDA[aux] = kernelCUDA[tid];//for the most part it should be a coalescent access in oth places
	  }
}


//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 
void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA)
{
	imageType* imCUDA = NULL;
	imageType* kernelCUDA = NULL;
	

	

	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	size_t imSize = 1;
	size_t kernelSize = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		imSize *= (imDim[ii]);
		kernelSize *= (kernelDim[ii]);
	}

	size_t imSizeFFT = imSize;
	imSizeFFT += 2*imDim[0]*imDim[1]; //size of the R2C transform in cuFFTComplex
	size_t imSizeFFTInByte = imSizeFFT*sizeof(imageType);
	size_t imSizeInByte = imSize*sizeof(imageType);
	size_t kernelSizeInByte = (kernelSize)*sizeof(imageType);
	//allocat ememory in GPU
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDA), imSizeFFTInByte ) );//a little bit larger to allow in-place FFT
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelCUDA), kernelSizeInByte ) );

	//printf("Copying memory (kernel and image) to GPU\n");
	HANDLE_ERROR( cudaMemcpy( kernelCUDA, kernel, kernelSizeInByte , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( imCUDA, im, imSizeInByte , cudaMemcpyHostToDevice ) );

	///////////////////////////////////////////////////////////////////////
	// DO THE WORK
	convolution3DfftCUDAInPlace_core(imCUDA, imDim,
					    kernelCUDA,kernelDim,
					    // &fftPlanFwd, &fftPlanInv,
					    devCUDA);
	///////////////////////////////////////////////////////////////////////

	//we destroy memory first so we can perform larger block convolutions

	

	//copy result to host and overwrite image
	HANDLE_ERROR(cudaMemcpy(im,imCUDA,sizeof(imageType)*imSize,cudaMemcpyDeviceToHost));

	
    //( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( imCUDA));
	HANDLE_ERROR( cudaFree( kernelCUDA));
	//HANDLE_ERROR( cudaFree( kernelCUDA));
	//HANDLE_ERROR( cudaFree( kernelPaddedCUDA));

}

//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 
void convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
					 imageType* _d_kernelCUDA,int* kernelDim,
					
					 int devCUDA)
{
  cufftHandle fftPlanFwd, fftPlanInv;
  imageType* kernelPaddedCUDA = NULL;
  //apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT

  size_t imSize = 1;
  size_t kernelSize = 1;
  for(int ii=0;ii<dimsImage;ii++)
    {
      imSize *= (imDim[ii]);
      kernelSize *= (kernelDim[ii]);
    }

  size_t imSizeFFT = imSize;
  imSizeFFT += 2*imDim[0]*imDim[1]; //size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT*sizeof(imageType);
  // size_t imSizeInByte = imSize*sizeof(imageType);
  // size_t kernelSizeInByte = (kernelSize)*sizeof(imageType);


  HANDLE_ERROR( cudaMalloc( (void**)&(kernelPaddedCUDA), imSizeFFTInByte ) );
  HANDLE_ERROR( cudaMemset( kernelPaddedCUDA, 0, imSizeFFTInByte ));

  size_t max_threads_on_device = getMaxNThreadsOfDevice(devCUDA);
  size_t max_blocks_in_x = getMaxNBlocksOfDevice(devCUDA,0); //we are using dim1 blocks only

  int numThreads=std::min( max_threads_on_device , kernelSize);
  size_t numBlocksFromImage = (kernelSize+numThreads-1)/(numThreads);
  int numBlocks=std::min(max_blocks_in_x,numBlocksFromImage);

  fftShiftKernel<<<numBlocks,numThreads>>>(_d_kernelCUDA,kernelPaddedCUDA,kernelDim[0],kernelDim[1],kernelDim[2],imDim[0],imDim[1],imDim[2]);HANDLE_ERROR_KERNEL;

	
  //make sure GPU finishes 
  HANDLE_ERROR(cudaDeviceSynchronize());	

  //printf("Creating R2C & C2R FFT plans for size %i x %i x %i\n",imDim[0],imDim[1],imDim[2]);
  cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C);HANDLE_ERROR_KERNEL;
  cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
  
  cufftExecR2C(fftPlanFwd, _d_imCUDA, (cufftComplex *)_d_imCUDA);HANDLE_ERROR_KERNEL;
  //transforming image
  cufftExecR2C(fftPlanFwd, kernelPaddedCUDA, (cufftComplex *)kernelPaddedCUDA);HANDLE_ERROR_KERNEL;

	
  size_t halfImSizeFFT = imSizeFFT/2;
  numThreads=std::min( max_threads_on_device , halfImSizeFFT);
  numBlocksFromImage = (halfImSizeFFT+numThreads-1)/(numThreads);
  numBlocks=std::min(max_blocks_in_x,numBlocksFromImage);

  
  modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)_d_imCUDA, 
							   (cufftComplex *)kernelPaddedCUDA, 
							   halfImSizeFFT,
							   1.0f/(float)(imSize));HANDLE_ERROR_KERNEL;//last parameter is the size of the FFT



  //inverse FFT 

  cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);HANDLE_ERROR_KERNEL;
  cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL;

  cufftExecC2R(fftPlanInv, (cufftComplex *)_d_imCUDA, _d_imCUDA);HANDLE_ERROR_KERNEL;
	
	
  HANDLE_ERROR( cudaFree( kernelPaddedCUDA));
  //release memory
  ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
  ( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
}

  
void compute_quotient(imageType* _input,imageType* _output,size_t _size, int _device){

  imageType* d_input = 0;
  imageType* d_output = 0;

  const size_t sizeInByte = _size*sizeof(imageType);
  HANDLE_ERROR(cudaSetDevice(_device));

  HANDLE_ERROR(cudaMalloc((void**)&d_input, sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_output, sizeInByte));
  
  HANDLE_ERROR(cudaMemcpy(d_input , _input , sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_output , _output , sizeInByte, cudaMemcpyHostToDevice));

 
  size_t items_per_block = 128;
  size_t items_per_grid = _size/items_per_block;
     
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 

  fit_2Dblocks_to_threads_for_device(threads,blocks,_device);

  //performs d_input_[:]=d_output_[:]/d_input_[:]
  device_divide<<<blocks,threads>>>(d_input, d_output, (unsigned int)_size);
  
  HANDLE_ERROR(cudaMemcpy(_output , d_output , sizeInByte, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(d_input));
  HANDLE_ERROR(cudaFree(d_output));
  
}

void compute_final_values(imageType* _image,
			  imageType* _integral,
			  imageType* _weight,
			  size_t _size,
			  float _minValue,
			  double _lambda, 
			  int _device){

  imageType* d_integral= 0;
  imageType* d_weight  = 0;
  imageType* d_image   = 0;

  const size_t sizeInByte = _size*sizeof(imageType);

  HANDLE_ERROR(cudaMalloc((void**)&d_integral, sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_weight  , sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_image   , sizeInByte));

  
  HANDLE_ERROR(cudaMemcpy(d_integral , _integral , sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_weight , _weight , sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_image , _image , sizeInByte, cudaMemcpyHostToDevice));
  
  size_t items_per_block = 128;
  size_t items_per_grid = _size/items_per_block;
    
    
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,_device);


  if(_lambda>0.)
    device_finalValues_tikhonov<<<blocks,threads>>>(d_image, d_integral, d_weight, _minValue, (float)_lambda, _size);
  else
    device_finalValues_plain<<<blocks,threads>>>(d_image, d_integral, d_weight, _minValue, _size);
  
  HANDLE_ERROR(cudaMemcpy(_image , d_image , sizeInByte, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(d_integral));
  HANDLE_ERROR(cudaFree(d_weight));
  HANDLE_ERROR(cudaFree(d_image));
  
}

void iterate_fft_plain(imageType* _input,
		       imageType* _kernel,
		       imageType* _output,
		       int* _input_dims,
		       int* _kernel_dims, 
		       int _device){
  
  size_t inputSize = std::accumulate(&_input_dims[0],&_input_dims[0]+dimsImage,1,multiplies<int>());
  size_t kernelSize = std::accumulate(&_kernel_dims[0],&_kernel_dims[0]+dimsImage,1,multiplies<int>());
  
  size_t inputInByte = inputSize*sizeof(imageType);
  size_t kernelInByte = kernelSize*sizeof(imageType);

  imageType* input = 0;
  imageType* kernel1 = 0;
  imageType* weights = 0;
  imageType* kernel2 = 0;
  HANDLE_ERROR( cudaMallocHost( (void**)&(kernel2),kernelInByte , cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaMallocHost( (void**)&(weights),inputInByte , cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaMallocHost( (void**)&(kernel1), kernelInByte, cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaMallocHost( (void**)&(input),inputInByte , cudaHostAllocDefault ) );

  std::fill(&kernel2[0],&kernel2[0]+kernelSize ,.1f);
  std::fill(&weights[0],&weights[0]+inputSize  ,1.f);
  std::copy(&_kernel[0], &_kernel[0] + kernelSize,&kernel1[0]);
  std::copy(&_input[0], &_input[0] + inputSize,&input[0]);

  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //
  imageType* d_image_ = 0;
  imageType* d_initial_ = 0;
  imageType* d_kernel_ = 0;
  imageType* d_weights_ = 0;

  size_t imSizeFFT = inputSize + 2*_input_dims[0]*_input_dims[1]; //size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT*sizeof(imageType);
  int gpu_device = selectDeviceWithHighestComputeCapability();

  HANDLE_ERROR( cudaMalloc( (void**)&(d_image_), imSizeFFTInByte ) );//a little bit larger to allow in-place FFT
  HANDLE_ERROR( cudaMalloc( (void**)&(d_initial_), inputInByte ) );
  HANDLE_ERROR( cudaMalloc( (void**)&(d_weights_), inputInByte ) );
  HANDLE_ERROR( cudaMalloc( (void**)&(d_kernel_), kernelInByte ) );
  cudaStream_t initial_stream, weights_stream;
  HANDLE_ERROR( cudaStreamCreate(&initial_stream));
  HANDLE_ERROR( cudaStreamCreate(&weights_stream));

  //TODO: should the weights be updated from device_divide (unclear in the java application)?
  HANDLE_ERROR( cudaMemcpyAsync( d_weights_, weights, inputInByte , cudaMemcpyHostToDevice, weights_stream ) );
  HANDLE_ERROR( cudaMemcpyAsync( d_initial_, input, inputInByte , cudaMemcpyHostToDevice ,initial_stream) );
  
  HANDLE_ERROR( cudaMemcpy( d_kernel_, kernel1, kernelInByte , cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( d_image_, input, inputInByte , cudaMemcpyHostToDevice ) );

  //convolve(input) with kernel1 -> psiBlurred
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims,
				   d_kernel_,_kernel_dims,
				   gpu_device);

  //computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = inputSize/items_per_block;
     
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,gpu_device);

  //performs d_initial_[:]=d_image_[:]/d_initial_[:]
  //TODO: should the weights be updated?
  device_divide<<<blocks,threads,0,initial_stream>>>(d_initial_,d_image_,inputSize);

  //convolve(psiBlurred) with kernel2 -> integral
  HANDLE_ERROR( cudaMemcpy( d_kernel_, kernel2, kernelInByte , cudaMemcpyHostToDevice ) );
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims,
				      d_kernel_,_kernel_dims,
				      gpu_device);
  //computeFinalValues(input,integral,weights)
  device_finalValues_plain<<<blocks,threads,0,weights_stream>>>(d_initial_,d_image_,d_weights_,.0001f,inputSize);
  
  
  HANDLE_ERROR(cudaMemcpyAsync(_output , d_initial_ , inputInByte , cudaMemcpyDeviceToHost, weights_stream));
  HANDLE_ERROR(cudaStreamSynchronize(weights_stream));
  
  HANDLE_ERROR( cudaFree( d_image_));
  HANDLE_ERROR( cudaFree( d_initial_));
  HANDLE_ERROR( cudaFree( d_kernel_));
  HANDLE_ERROR( cudaFree( d_weights_));
  
  HANDLE_ERROR( cudaFreeHost( kernel2));
  HANDLE_ERROR( cudaFreeHost( weights));
  HANDLE_ERROR( cudaFreeHost( kernel1));
  HANDLE_ERROR( cudaFreeHost( input  ));
  HANDLE_ERROR( cudaStreamDestroy(initial_stream));
  HANDLE_ERROR( cudaStreamDestroy(weights_stream));
	 
}

void iterate_fft_tikhonov(imageType* _input,
					imageType* _kernel,
					imageType* _output,
					int* _input_dims,
					int* _kernel_dims,
					size_t _size, 
					float _minValue, 
					double _lambda, 
					int _device){
  
  size_t inputSize = std::accumulate(&_input_dims[0],&_input_dims[0]+dimsImage,1,multiplies<int>());
  size_t kernelSize = std::accumulate(&_kernel_dims[0],&_kernel_dims[0]+dimsImage,1,multiplies<int>());
  
  size_t inputInByte = inputSize*sizeof(imageType);
  size_t kernelInByte = kernelSize*sizeof(imageType);

  std::vector<imageType>* kernel2_ = new std::vector<imageType>(kernelSize);
  std::vector<imageType>* weights_ = new std::vector<imageType>(inputSize);
  std::fill(kernel2_->begin(),kernel2_->end(),.1f);
  std::fill(weights_->begin(),weights_->end(),1.f);
    
  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //
  imageType* d_image_ = 0;
  imageType* d_initial_ = 0;
  imageType* d_kernel_ = 0;
  imageType* d_weights_ = 0;

  size_t imSizeFFT = inputSize + 2*_input_dims[0]*_input_dims[1]; //size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT*sizeof(imageType);
  int gpu_device = selectDeviceWithHighestComputeCapability();

  HANDLE_ERROR( cudaMalloc( (void**)&(d_image_), imSizeFFTInByte ) );//a little bit larger to allow in-place FFT
  HANDLE_ERROR( cudaMalloc( (void**)&(d_initial_), inputInByte ) );
  HANDLE_ERROR( cudaMalloc( (void**)&(d_weights_), inputInByte ) );
  HANDLE_ERROR( cudaMalloc( (void**)&(d_kernel_), kernelInByte ) );
  HANDLE_ERROR( cudaMemcpy( d_weights_, &weights_[0], inputInByte , cudaMemcpyHostToDevice ) );

  

  HANDLE_ERROR( cudaMemcpy( d_kernel_, _kernel, kernelInByte , cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( d_image_, _input, inputInByte , cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( d_initial_, d_image_, inputInByte , cudaMemcpyDeviceToDevice ) );

  
  //convolve(input) with kernel1 -> psiBlurred
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims,
				      d_kernel_,_kernel_dims,
				      gpu_device);

  //computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = inputSize/items_per_block;
     
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,gpu_device);

  //performs d_initial_[:]=d_image_[:]/d_initial_[:]
  //TODO: should the weights be updated?
  device_divide<<<blocks,threads>>>(d_initial_,d_image_,inputSize);

  //convolve(psiBlurred) with kernel2 -> integral
  HANDLE_ERROR( cudaMemcpy( d_kernel_, &kernel2_[0], kernelInByte , cudaMemcpyHostToDevice ) );
  convolution3DfftCUDAInPlace_core(d_image_, _input_dims,
				      d_kernel_,_kernel_dims,
				      gpu_device);
  //computeFinalValues(input,integral,weights)
  device_finalValues_tikhonov<<<blocks,threads>>>(d_initial_,d_image_,d_weights_,.0001f,.2f,inputSize);
 
  
  HANDLE_ERROR(cudaMemcpy(_output , d_initial_ , inputInByte , cudaMemcpyDeviceToHost));
 
  HANDLE_ERROR( cudaFree( d_image_));
  HANDLE_ERROR( cudaFree( d_initial_));
  HANDLE_ERROR( cudaFree( d_kernel_));
  HANDLE_ERROR( cudaFree( d_weights_));

  delete kernel2_;
  delete weights_;


}
